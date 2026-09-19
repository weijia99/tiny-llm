from typing import Any

import mlx.core as mx

from extensions import tiny_llm_ext


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w.astype(mx.bfloat16)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self.use_simdgroup_matmul = use_simdgroup_matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec
        self.use_split_k_matmul = use_split_k_matmul
        self.use_mlx_quantized_linear = use_mlx_quantized_linear

    @staticmethod
    def from_mlx_layer(
        mlx_layer: Any,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ) -> "QuantizedWeights":
        biases = mlx_layer.biases
        return QuantizedWeights(
            scales=mlx_layer.scales.astype(mx.bfloat16),
            biases=None if biases is None else biases.astype(mx.bfloat16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
            use_simdgroup_matmul=use_simdgroup_matmul,
            use_simdgroup_matvec=use_simdgroup_matvec,
            use_split_k_matmul=use_split_k_matmul,
            use_mlx_quantized_linear=use_mlx_quantized_linear,
        )


def mlx_quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    pass


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> mx.array:
    # 实现转发到对应的C++接口
    return tiny_llm_ext.quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        use_simdgroup,
        use_split_k,
    )


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    # 实现对应的dequantization操作
    # 输出dense_weigts [k,N] BF16
    # scale [k,N/group_size] BF16
    # bias [k,N/group_size] BF16
    # 通过向量乘法来进行加速计算，实现ax+b的操作
    # 1.weights 中的一个uint32等于8个uint4，8个uint4对应8个BF16的scale和bias
    # 2.需要解开对应的uint32，变成对应的uint4，然后进行广播
    shifts = mx.arange(32//bits, dtype=mx.int32)*bits
    # 生成对应的mask0，4，8，12，16，20，24，28，相当于解开了32位的uint32，变成了8个uint4
    x = (weight[..., None] >> shifts) & ((1 << bits) - 1)
    x = x.reshape(*x.shape[:-2], x.shape[-2]*x.shape[-1])
    # 左边形状 (..., N/8, 1)，右边形状 (8,)。按广播规则，最后两维对齐后扩展成 (..., N/8, 8)
    # 最终x的形状为 (..., N/8, 8)
    # 接下来实现scale*x + bias的操作
    #开始计算重复次数
    repeat_times = group_size
    # 广播重复
    scale = mx.repeat(scales[..., None], repeat_times, axis=-1)
    scale = scale.reshape(x.shape)
    if biases is not None:
        bias = mx.repeat(biases[..., None], repeat_times, axis=-1)
        bias = bias.reshape(x.shape)
        return (x.astype(mx.bfloat16) * scale + bias).astype(mx.bfloat16)
    else:
        return (x.astype(mx.bfloat16) * scale).astype(mx.bfloat16)



def quantized_matvec_custom(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return tiny_llm_ext.quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        True,
    )


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return tiny_llm_ext.quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        False
    )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    # 实现解包后的线性层计算
    # 1.解包weights
    scale = w.scales
    bias_w = w.biases
    group_size = w.group_size
    bits = w.bits
    weight = w.weight
    # 2.解包weights
    # dequantized_weight = dequantize_weights(weight, scale, bias_w, group_size, bits)
    # 3.计算线性层
    # output = mx.matmul(x, dequantized_weight.T)
    # 更新实现对应的c++实现
    output  = quantized_matmul(
        scales=scale,
        biases=bias_w,
        group_size=group_size,
        bits=bits,
        a=x,
        b=weight,
        transpose_b=True,
        use_simdgroup=w.use_simdgroup_matmul,
        use_split_k=w.use_split_k_matmul,
    )
    if bias is not None:
        output = output + bias
    return output
