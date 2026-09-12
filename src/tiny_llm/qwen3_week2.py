from typing import Any

import mlx.core as mx

from .attention import scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding
from .kv_cache import TinyKvCache, TinyKvFullCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, dequantize_linear
from .week2_kernels import FastRMSNorm, FastRoPE, swiglu

WEEK2_CHECKPOINTS = (
    "kv-cache",
    "quantized-matvec",
    "rmsnorm",
    "rope",
    "swiglu",
    "decode-attention",
    "simd-matmul",
    "split-k",
)

DECODE_ATTENTION_MAX_CONTEXT = 256
DECODE_ATTENTION_MAX_QUERY = 2


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_decode_attention: bool = True,
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
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_decode_attention = use_decode_attention
        self.rope = RoPE(self.head_dim, self.max_seq_len, self.theta)

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        q = linear(x, self.wq).reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        k = linear(x, self.wk).reshape(
            *x.shape[:-1], self.num_kv_heads, self.head_dim
        )
        v = linear(x, self.wv).reshape(
            *x.shape[:-1], self.num_kv_heads, self.head_dim
        )
        rms_norm_q = RMSNorm(self.head_dim, self.q_norm, self.rms_norm_eps)
        rms_norm_k = RMSNorm(self.head_dim, self.k_norm, self.rms_norm_eps)
        q = rms_norm_q(q)
        k = rms_norm_k(k)
        q = self.rope(q, slice(offsets, offsets + x.shape[-2]))
        k = self.rope(k, slice(offsets, offsets + x.shape[-2]))
        q = q.swapaxes(-2, -3)
        k = k.swapaxes(-2, -3)
        v = v.swapaxes(-2, -3)
        k, v = cache.update_and_fetch(
            k, v, mask_length=x.shape[-2], mask=mask
        )[:2]
        output = scaled_dot_product_attention_grouped(q, k, v, mask=mask).astype(x.dtype)
        output = output.swapaxes(-2, -3).reshape(
            *x.shape[:-1], self.num_heads * self.head_dim
        )
        return linear(output, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_decode_attention: bool = True,
    ):
        self.num_kv_heads = num_kv_heads
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_fast_swiglu = use_fast_swiglu
        self.use_decode_attention = use_decode_attention
        self.input_layernorm = RMSNorm(
            self.hidden_size, self.w_input_layernorm, self.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, self.w_post_attention_layernorm, self.rms_norm_eps
        )

        self.attention = Qwen3MultiHeadAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            self.q_norm,
            self.k_norm,
            self.max_seq_len,
            self.theta,
            self.rms_norm_eps,
            self.use_fast_rms_norm,
            self.use_fast_rope,
            self.use_decode_attention,
        )
        self.self_attn = self.attention
        self.mlp = Qwen3MLP(
            self.hidden_size,
            self.intermediate_size,
            self.w_gate,
            self.w_up,
            self.w_down,
            self.use_fast_swiglu,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_norm = self.input_layernorm(x)
        attention_output = self.attention(input_norm, offset, cache, mask=mask)
        attention_output = attention_output + x
        post_attention_norm = self.post_attention_layernorm(attention_output)
        mlp_output = self.mlp(post_attention_norm)
        return mlp_output + attention_output


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        checkpoint: str = "split-k",
        use_mlx_quantized_linear: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.num_attention_heads = mlx_model.args.num_attention_heads
        self.num_kv_heads = mlx_model.args.num_key_value_heads
        self.head_dim = mlx_model.args.head_dim
        self.intermediate_size = mlx_model.args.intermediate_size
        self.rms_norm_eps = mlx_model.args.rms_norm_eps
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.embedding = Embedding(
            mlx_model.args.vocab_size,
            mlx_model.args.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens),
        )
        self.precision = mx.bfloat16
        self.checkpoint = checkpoint
        self.mlx_model = mlx_model
        use_fast_kernels = checkpoint != "kv-cache"
        self.layers_inner = [
            Qwen3TransformerBlock(
                self.num_attention_heads,
                self.num_kv_heads,
                self.hidden_size,
                self.head_dim,
                self.intermediate_size,
                self.rms_norm_eps,
                dequantize_linear(self.mlx_model.model.layers[i].self_attn.q_proj),
                dequantize_linear(self.mlx_model.model.layers[i].self_attn.k_proj),
                dequantize_linear(self.mlx_model.model.layers[i].self_attn.v_proj),
                dequantize_linear(self.mlx_model.model.layers[i].self_attn.o_proj),
                self.mlx_model.model.layers[i].self_attn.q_norm.weight,
                self.mlx_model.model.layers[i].self_attn.k_norm.weight,
                dequantize_linear(self.mlx_model.model.layers[i].mlp.gate_proj),
                dequantize_linear(self.mlx_model.model.layers[i].mlp.up_proj),
                dequantize_linear(self.mlx_model.model.layers[i].mlp.down_proj),
                self.mlx_model.model.layers[i].input_layernorm.weight,
                self.mlx_model.model.layers[i].post_attention_layernorm.weight,
                self.mlx_model.args.max_position_embeddings,
                self.mlx_model.args.rope_theta,
                use_fast_rms_norm=use_fast_kernels,
                use_fast_rope=use_fast_kernels,
                use_fast_swiglu=use_fast_kernels,
                use_decode_attention=use_fast_kernels,
            )
            for i in range(self.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )
        self.w_lm_head = None

    def create_kv_cache(self) -> list[TinyKvCache]:
        return [TinyKvFullCache() for _ in range(self.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        for layer, layer_cache in enumerate(cache):
            if layer_cache.offset != offset:
                raise ValueError(
                    f"layer {layer} cache offset {layer_cache.offset} "
                    f"does not match model offset {offset}"
                )
        x = self.embedding(inputs)
        for i in range(self.num_hidden_layers):
            x = self.layers_inner[i](x, offset=offset, cache=cache[i], mask="causal")
        x = self.norm(x)

        if self.w_lm_head is not None:
            return linear(x, self.w_lm_head)
        return self.embedding.as_linear(x)
