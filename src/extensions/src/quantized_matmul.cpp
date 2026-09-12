#include <mlx/dtype.h>
#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 3. Days 6 and 7 extend the dispatch policy behind this API.
mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, int group_size, int bits, const mx::array &a, const mx::array &b, bool transpose_b,
                           bool use_simdgroup, bool use_split_k, mx::StreamOrDevice stream) {
    // checkpoint_todo("quantized_matmul", "Week 2, Day 3");
    // 需要实现这个接口 
    //主要功能就是实现对应layzoutput，只需要校验assert判断
    // stream 定位到是哪一个device上面去执行
    // 1. 校验输入的参数是否符合要求
    // 2. 然后根据对应的参数，生成需要output的shape数组
    // dtype
    // ---------- 1. 参数校验 ----------
  if (bits <= 0 || 32 % bits != 0) {
    throw std::invalid_argument("[quantized_matmul] bits must divide 32 (2, 4, 8).");
  }
  const int pack = 32 / bits;                      // 一个 uint32 装几个权重
  if (group_size <= 0 || group_size % pack != 0) {
    throw std::invalid_argument("[quantized_matmul] group_size must be a positive multiple of 32/bits.");
  }
  if (a.ndim() < 2 || b.ndim() < 2) {
    throw std::invalid_argument("[quantized_matmul] a and b must be at least 2D.");
  }
  if (b.dtype() != mx::uint32) {
    throw std::invalid_argument("[quantized_matmul] packed weights must be uint32.");
  }
  if(a.dtype() !=mx::bfloat16){
    throw std::invalid_argument("[quantized_matmul] a must be bfloat16.");
  }
  if(scales.dtype() != mx::bfloat16){
    throw std::invalid_argument("[quantized_matmul] scales must be bfloat16.");
  }
  if(biases.dtype() != mx::bfloat16){
    throw std::invalid_argument("[quantized_matmul] biases must be bfloat16.");
  }


  if (scales.shape() != biases.shape()) {
    throw std::invalid_argument("[quantized_matmul] scales and biases must share a shape.");
  }

   // ---------- 2. 形状推导 ----------
  const auto &as = a.shape();
  const auto &bs = b.shape();
  const auto &ss = scales.shape();
  const auto &biass = biases.shape();

  const int N        = as[as.size() - 1];                                   // 收缩维 in_features
  //是否使用了转置
  const int K        = transpose_b ? bs[bs.size() - 2] : bs[bs.size() - 1]; // out_features
  const int packed_n = transpose_b ? bs[bs.size() - 1] : bs[bs.size() - 2]; // 打包维

  if (static_cast<long>(packed_n) * pack != N) {
    throw std::invalid_argument("[quantized_matmul] in_features mismatch between a and b.");
  }
  if ( ss[ss.size() - 2] != K or biass[biass.size() - 2] != K) {
    throw std::invalid_argument("[quantized_matmul] scales must be (out_features, in_features/group_size).");
  }
  if (static_cast<long>(ss[ss.size() -1]) * group_size != N) {
    throw std::invalid_argument("[quantized_matmul] group_size does not tile in_features.");
  }
  if (static_cast<long>(biass[biass.size() -1])*group_size != N) {
    throw std::invalid_argument("[quantized_matmul] biases must be (out_features, in_features/group_size).");
  }
  //生成对应的shape
  mx::Shape output_shape = a.shape();   // (..., M, N)
  output_shape.back() = K;              // (..., M, K)
  auto output_dtype = promote_types(a.dtype(), scales.dtype());
  //传入对应的参数
  auto primitive = std::make_shared<QuantizedMatmul>(
    to_stream(stream),
    use_simdgroup,
    use_split_k
);

    return mx::array(output_shape, output_dtype, std::move(primitive),
                   /* inputs = */ {a, b, scales, biases});

}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    // checkpoint_todo("QuantizedMatmul::eval_cpu", "Week 2, Day 3");
    throw std::runtime_error("QuantizedMatmul has no CPU implementation.");
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedMatmul::eval_gpu", "Week 2, Day 3");
}

// Week 3, Day 4. The earlier Week 2 checkpoints keep the readable row lookup.
mx::array quantized_embedding(const mx::array &, const mx::array &, const mx::array &, const mx::array &, int, int,
                              mx::StreamOrDevice) {
    checkpoint_todo("quantized_embedding", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_cpu", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
