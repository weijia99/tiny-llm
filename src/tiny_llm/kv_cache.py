from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import mlx.core as mx

if TYPE_CHECKING:
    from .paged_kv_cache import PagedKvMetadata


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            The updated keys, updated values, sequence length, and mask. On
            On Week 2 Day 1, the mask is passed through unchanged. Week 3 Day 1
            uses the sequence length and mask to construct a dense batch.
        """

    def release(self):
        pass

    def materialize(self):
        """Evaluate owned K/V storage without changing its logical layout."""
        pass

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        pass

    def rewind(self, n: int):
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int | None = None):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self, capacity: int | None = None):
        if capacity is not None and (
            not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0
        ):
            raise ValueError("capacity must be a non-negative integer or None")
        self.key_values = None
        self.offset = 0
        self.capacity = capacity
        self.logical_copy_bytes = 0
        self.physical_growth_copy_bytes = 0
        self.slice_write_bytes = 0
        self.growth_copy_bytes = 0

    @property
    def uses_capacity(self) -> bool:
        return self.capacity is not None

    def _logical_key_values(self) -> tuple[mx.array, mx.array]:
        """Return initialized tokens without the unused capacity tail."""
        keys, values = self.key_values
        return keys[:, :, : self.offset], values[:, :, : self.offset]

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if key.shape != value.shape:
            raise ValueError("key and value chunks must have the same shape")
        batch, heads, sequence, head_dim = key.shape
        if self.uses_capacity:
            end = self.offset + sequence
            if end > self.capacity:
                raise ValueError(
                    f"KV cache capacity {self.capacity} exceeded by append ending at {end}"
                )
            if self.key_values is None:
                keys = mx.zeros((batch, heads, self.capacity, head_dim), dtype=key.dtype)
                values = mx.zeros((batch, heads, self.capacity, head_dim), dtype=value.dtype)
                self.key_values = (keys, values)
            else:
                keys, values = self.key_values
                if keys.shape != (batch, heads, self.capacity, head_dim):
                    raise ValueError("KV cache shape changed during a request")
            if sequence:
                start = mx.array([self.offset])
                keys, values = self.key_values
                self.key_values = (
                    mx.slice_update(keys, key, start_indices=start, axes=(2,)),
                    mx.slice_update(values, value, start_indices=start, axes=(2,)),
                )
                self.slice_write_bytes += key.nbytes + value.nbytes
            self.offset = end
            keys, values = self._logical_key_values()
            return keys, values, self.offset, mask

        if self.key_values is None:
            self.key_values = (key, value)
            self.offset = sequence
            return key, value, self.offset, mask
        keys, values = self.key_values
        copied_bytes = keys.nbytes + values.nbytes
        self.logical_copy_bytes += copied_bytes
        self.physical_growth_copy_bytes += copied_bytes
        self.growth_copy_bytes += copied_bytes
        self.key_values = (
            mx.concat([keys, key], axis=2),
            mx.concat([values, value], axis=2),
        )
        self.offset += sequence
        keys, values = self.key_values
        return keys, values, self.offset, mask

    def materialize(self):
        if self.key_values is not None:
            mx.eval(*self.key_values)

    def reset(self):
        self.offset = 0
        if not self.uses_capacity:
            self.key_values = None

    def rewind(self, n: int):
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= self.offset:
            raise ValueError("rewind length must be between zero and the cache length")
        self.offset -= n
        if self.offset == 0 and not self.uses_capacity:
            self.key_values = None
        elif not self.uses_capacity:
            keys, values = self.key_values
            self.key_values = (keys[:, :, : self.offset], values[:, :, : self.offset])
