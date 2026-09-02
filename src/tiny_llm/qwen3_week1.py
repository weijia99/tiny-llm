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
        # q = mx.fast.rms_norm(q,self.q_norm,self.rms_norm_eps)
        # k = mx.fast.rms_norm(k,self.k_norm,self.rms_norm_eps)
        rms_norm_q = RMSNorm(self.head_dim, self.q_norm, self.rms_norm_eps)
        rms_norm_k = RMSNorm(self.head_dim, self.k_norm, self.rms_norm_eps)
        q = rms_norm_q(q)
        k = rms_norm_k(k)
        rope = RoPE(self.head_dim, self.max_seq_len, self.theta)
        q = rope(q, slice(0, x.shape[-2]))   # 应用，不是构造
        k = rope(k, slice(0, x.shape[-2]))
        q = q.swapaxes(-2, -3)  # (batch, head, seq_len, head_dim)
        k = k.swapaxes(-2, -3)  # (batch, head,
        v = v.swapaxes(-2, -3)  # (batch, head, seq_len, head_dim)
        output = scaled_dot_product_attention_grouped(
            q, k, v, mask=mask
        ).swapaxes(-2, -3).reshape(*x.shape[:-1], self.num_heads * self.head_dim)
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
        self.rms_norml = RMSNorm(self.hidden_size, self.w_input_layernorm, self.rms_norm_eps)
        self.rms_norm2 = RMSNorm(self.hidden_size, self.w_post_attention_layernorm, self.rms_norm_eps)
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
        )
        self.mlp = Qwen3MLP(
            self.hidden_size,
            self.intermediate_size,
            self.w_gate,
            self.w_up,
            self.w_down,
        )


    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_norm = self.rms_norml(x)
        attention_output = self.attention(input_norm, mask=mask)
        attention_output = attention_output + x
        post_attention_norm = self.rms_norm2(attention_output)
        mlp_output = self.mlp(post_attention_norm)
        output = mlp_output + attention_output
        return output
        


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        
        precision = mx.bfloat16
        self.precision = precision
        # 量化embedding和attn
        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens),
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            layer = Qwen3TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                head_dim=mlx_model.args.head_dim,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj),
                wk=dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj),
                wv=dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj),
                wo=dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj),
                q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_gate=dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj),
                w_up=dequantize_linear(mlx_model.model.layers[i].mlp.up_proj),
                w_down=dequantize_linear(mlx_model.model.layers[i].mlp.down_proj),
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model



    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        x = self.embedding(inputs)
        for i in range(self.num_hidden_layers):
            x = self.layers_inner[i](x,mask="causal")
        x = self.norm(x)
        
        if self.w_lm_head is not None:
            return linear(x, self.w_lm_head)
        else:
            return self.embedding.as_linear(x)
