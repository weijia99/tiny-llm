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

// Vanilla path: one thread computes one output element out[row, k].
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
    uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint k = gid.y;

    if (row >= M || k >= K) {
        return;
    }

    float sum = 0.0f;
    for (uint j = 0; j < N; ++j) {
        uint packed_weight = packed_weights[k * (N / 8) + (j / 8)];
        uint index = j % 8;
        uint8_t weight = (packed_weight >> (index * 4)) & 0xF;
        float scale_weight = float(scales[k * (N / 128) + (j / 128)]);
        float bias_weight = float(biases[k * (N / 128) + (j / 128)]);
        float W = weight * scale_weight + bias_weight;
        float A = float(activations[row * N + j]);
        sum += W * A;
    }

    out[row * K + k] = T(sum);
}

// SIMD path: two SIMD groups per threadgroup; each group calculates four
// consecutive output columns while its 32 lanes split packed-weight words.
template <typename T>
[[kernel]] void quantized_matvec_x4_fast_w4a16_g128(
    device const T* activations [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const T* scales [[buffer(2)]],
    device const T* biases [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint outputs_per_threadgroup = 8;
    constexpr uint outputs_per_simdgroup = 4;
    constexpr uint packed_values = 8;
    constexpr uint packed_words_per_group = 16;

    const uint column_tiles = (K + outputs_per_threadgroup - 1) /
                              outputs_per_threadgroup;
    const uint row = output_tile / column_tiles;
    const uint column_tile = output_tile % column_tiles;
    const uint k_base = column_tile * outputs_per_threadgroup +
                        simdgroup * outputs_per_simdgroup;

    if (row >= M || k_base >= K) {
        return;
    }

    const uint packed_cols = N / packed_values;
    const uint groups_per_row = N / 128;
    float partial_sums[outputs_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint packed_col = lane; packed_col < packed_cols;
         packed_col += 32) {
        const uint group = packed_col / packed_words_per_group;
        const uint a_base = packed_col * packed_values;

        float activation_pack[packed_values];
        for (uint value = 0; value < packed_values; ++value) {
            activation_pack[value] = float(
                activations[row * N + a_base + value]);
        }

        for (uint output = 0; output < outputs_per_simdgroup; ++output) {
            const uint k = k_base + output;
            if (k >= K) {
                continue;
            }

            const uint packed = packed_weights[k * packed_cols + packed_col];
            const float scale = float(scales[k * groups_per_row + group]);
            const float bias = float(biases[k * groups_per_row + group]);
            for (uint value = 0; value < packed_values; ++value) {
                const float weight = float((packed >> (value * 4)) & 0xF) *
                                     scale + bias;
                partial_sums[output] += activation_pack[value] * weight;
            }
        }
    }

    for (uint output = 0; output < outputs_per_simdgroup; ++output) {
        const uint k = k_base + output;
        const float total_sum = simd_sum(partial_sums[output]);
        if (lane == 0 && k < K) {
            out[row * K + k] = T(total_sum);
        }
    }
}

instantiate_kernel(
    "quantized_matmul_vanilla_w4a16_g128_bfloat16",
    quantized_matmul_vanilla_w4a16_g128,
    bfloat16_t
);

instantiate_kernel(
    "quantized_matvec_x4_fast_w4a16_g128_bfloat16",
    quantized_matvec_x4_fast_w4a16_g128,
    bfloat16_t
);