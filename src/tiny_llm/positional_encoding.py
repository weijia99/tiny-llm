import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        
        self.dims = dims
        self.seq_len = seq_len
        self.half_dims = dims // 2
        self.base = base
        self.traditional = traditional
        self.cos_cache = mx.zeros((seq_len, self.half_dims), dtype=mx.float32)
        self.sin_cache = mx.zeros((seq_len, self.half_dims), dtype=mx.float32)

        freqs = base ** (-2 * mx.arange(self.half_dims, dtype=mx.float32) / dims)
        # for pos in range(seq_len):
        #     self.cos_cache[pos] = mx.cos(pos * freqs)
        #     self.sin_cache[pos] = mx.sin(pos * freqs)
        # 向量优化
        self.cos_cache = mx.cos(mx.arange(seq_len, dtype=mx.float32).reshape(-1, 1) * freqs)
        self.sin_cache = mx.sin(mx.arange(seq_len, dtype=mx.float32).reshape(-1, 1) * freqs)


    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        assert D == self.dims, f"expected head dimension {self.dims}, got {D}"
        if offset is None:
            # 直接取序列的前L个位置的cos和sin
            cos = self.cos_cache[:L]
            sin = self.sin_cache[:L]
        elif isinstance(offset, slice):
            assert offset.stop - offset.start == L, (
                f"offset must be of length {L}"
            )
            cos = self.cos_cache[offset]
            sin = self.sin_cache[offset]
        else:
            raise ValueError("Week 1 RoPE only supports a single slice offset")

        # Broadcast the basis over the batch and head dimensions.
        cos = cos.reshape(1, L, 1, self.half_dims)
        sin = sin.reshape(1, L, 1, self.half_dims)

        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
            real = x1 * cos - x2 * sin
            imag = x2 * cos + x1 * sin
            output = mx.stack([real, imag], axis=-1).reshape(N, L, H, D)
        else:
            x1 = x[..., : self.half_dims]
            x2 = x[..., self.half_dims :]
            real = x1 * cos - x2 * sin
            imag = x2 * cos + x1 * sin
            output = mx.concat([real, imag], axis=-1).reshape(N, L, H, D)

        return output.astype(x.dtype)
