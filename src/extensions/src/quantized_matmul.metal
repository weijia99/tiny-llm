#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
using namespace metal;

// Starter interface map. Implement the named kernels at these checkpoints;
// their argument lists are defined by the matching C++ encoder you complete.
//
// Week 2, Day 3:
//   quantized_matmul_vanilla_w4a16_g128
//   quantized_matvec_x4_fast_w4a16_g128
// Week 2, Day 6:
//   quantized_matmul_simdgroup_w4a16_g128
// Week 2, Day 7:
//   quantized_matmul_simdgroup_splitk_w4a16_g128
//   quantized_matmul_splitk_reduce
// Week 3, Day 4:
//   quantized_embedding_w4a16_g128
//
// The x2/x8 tuning variants in the reference extension are deliberately not
// starter interfaces. Add an experimental variant only while running the
// optional scheduling comparison, then keep the selected course path.

//实现骨骼模板
template <typename T>
[[kernel]] void quantized_matmul_vanilla_w4a16_g128(
    device const T* activations [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const T* scales [[buffer(2)]],
    device const T* biases [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]

){
    //从一个thread获取到i，k
    //进行解码uint8，然后回复，之后与activatare相乘，写出out
    uint row = gid.x;
    uint k =gid.y;
    //检查是否越界
     if (row >= M || k >= K) {
        return;
    }

    //metal中没有二维数组，所以需要手动计算索引

    //这边就是实现out的i，j的计算
    float sum = 0.0f;
    for(uint j =0;j<N;j++){
        //解码int4
        uint packed_weight = packed_weights[k * (N/8) + (j/8)]; 
        //获取索引
        uint index = j%8;
        //获取对应的值
        uint8_t weight = (packed_weight >> (index * 4)) & 0xF;
        //最终是要叠加N次作为计算和的结果
        float scale_weight = float(scales[k * (N/128) + (j/128)]);
        float bias_weight = float(biases[k * (N/128) + (j/128)]);
        //将权重转换为浮点数
        float W = weight * scale_weight + bias_weight;
        //找出对应的激活值A
        float A = float(activations[row * N + j]);
        //累加到sum中
        sum += W * A;
    }
    out[row * K + k] = T(sum);
}


//映射到metal中
instantiate_kernel(
    "quantized_matmul_vanilla_w4a16_g128_bfloat16",
    quantized_matmul_vanilla_w4a16_g128,
    bfloat16_t
);