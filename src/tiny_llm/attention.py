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
    # mask = mx.full((L,S), -mx.inf, dtype=dtype)
    # dis = S - L if S > L else 0
    # for i in range(L):
    #     for j in range(dis+i+1):
    #         mask[i,j] = 0
    # return mask
    # 生成下三角矩阵
    dis = S - L if S > L else 0
    return mx.triu(mx.full((L, S), -mx.inf, dtype=dtype), k=dis + 1)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    # 抽取出hidden和repeat的维度
    query_head = query.shape[-3]
    key_head = key.shape[-3]
    n_repeat = query_head // key_head
    # 计算缩放因子
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    query = query.reshape(*query.shape[:-3], key_head, n_repeat,query.shape[-2], query.shape[-1])
    key = key.reshape(*key.shape[:-3], key_head, 1, key.shape[-2], key.shape[-1])
    value = value.reshape(*value.shape[:-3], key_head, 1, value.shape[-2], value.shape[-1])
    # 计算qkt
    hidden = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    # 处理mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            mask = causal_mask(hidden.shape[-2], hidden.shape[-1], hidden.dtype)
            hidden = hidden + mask
        else:
            mask = mask.reshape(*hidden.shape[:-4], key_head, n_repeat, hidden.shape[-2], hidden.shape[-1])
            hidden = hidden + mask
    # 计算softmax
    hidden = softmax(hidden, axis=-1)
    # 计算qktv
    hidden = mx.matmul(hidden, value)
    # 恢复原来的维度
    hidden = hidden.reshape(*query.shape[:-4], query_head, hidden.shape[-2], hidden.shape[-1])
    return hidden


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
