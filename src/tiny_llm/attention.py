import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # 0。特殊处理
    # 1.计算qkt
    # 2.实现缩放效果
    # 3.实现mask效果
    # 4.实现softmax
    # 5.实现qktv
   
    # 传了就用，没传自己进行求平方根
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    hidden = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        hidden = hidden + mask
    hidden = softmax(hidden,axis=-1)
    hidden = mx.matmul(hidden,value)
    return hidden


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        p_q = linear(query, self.wq).reshape(*query.shape[:-1], self.num_heads, self.head_size).swapaxes(-2, -3)
        p_k = linear(key, self.wk).reshape(*key.shape[:-1], self.num_heads, self.head_size).swapaxes(-2, -3)
        p_v = linear(value, self.wv).reshape(*value.shape[:-1], self.num_heads, self.head_size).swapaxes(-2, -3)
        scaled_attention = scaled_dot_product_attention_simple(p_q, p_k, p_v, mask=mask).swapaxes(-2, -3).reshape(*query.shape[:-1], self.hidden_size)
        output = linear(scaled_attention, self.wo)
        return output
        


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
