import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 最终是head，L，dimension
        q = linear(x, self.wq).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        k = linear(x, self.wk).reshape(*x.shape[:-1], self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv).reshape(*x.shape[:-1], self.num_kv_heads, self.head_dim)
        q = mx.fast.rms_norm(q,self.q_norm,self.rms_norm_eps)
        k = mx.fast.rms_norm(k,self.k_norm,self.rms_norm_eps)
        rope = RoPE(self.head_dim, self.max_seq_len, self.theta)
        q = rope(q, slice(0, x.shape[-2]))   # 应用，不是构造
        k = rope(k, slice(0, x.shape[-2]))
        q = q.swapaxes(-2, -3)  # (batch, head, seq_len, head_dim)
        k = k.swapaxes(-2, -3)  # (batch, head,
        v = v.swapaxes(-2, -3)  # (batch, head, seq_len, head_dim)
        output = scaled_dot_product_attention_grouped(
            q, k, v, mask=mask
        ).swapaxes(-2, -3).reshape(*x.shape[:-1], self.hidden_size)
        output = linear(output, self.wo)
        return output



class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
