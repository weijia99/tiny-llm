import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    return mx.matmul(x, w.T) + bias if bias is not None else mx.matmul(x, w.T)


def silu(x: mx.array) -> mx.array:
    z = mx.exp(-abs(x))
    # 通过使用where进行判断条件，那些大于等于0的元素使用x/(1+z)，小于0的元素使用x*z/(1+z)
    ans = mx.where(x >= 0, x / (1 + z), x * z / (1 + z))
    return ans
