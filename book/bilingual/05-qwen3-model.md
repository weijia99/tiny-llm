# Day 5: The Qwen3 Model

# 第 5 天：Qwen3 模型

On Day 5, we will combine the components from the previous chapters into the complete Qwen3 model.

zh> 第 5 天，我们将把前几章实现的各个组件组合成完整的 Qwen3 模型。

Model-level tests require the corresponding model files. Start with the default 0.6B model; download the larger models
only if you want to test them as well:

zh> 模型级别的测试需要对应的模型文件。请先从默认的 0.6B 模型开始；更大的模型只在你也想测试它们时才下载：

```bash
hf download Qwen/Qwen3-0.6B-MLX-4bit
# Optional larger models:
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
```

Tests that require an unavailable model will be skipped.

zh> 需要某个未下载模型的测试会被自动跳过。

## Task 1: Implement `Qwen3TransformerBlock`

## 任务 1：实现 `Qwen3TransformerBlock`

```
src/tiny_llm/qwen3_week1.py
```

**📚 延伸阅读 / Readings**

- [A Simplified Explanation of the Transformer Block](https://medium.com/@akhileshkapse/a-simplified-explanation-of-the-transformer-block-must-read-blog-for-nlp-enthusiasts-12ef240a62ac)
- [Attention is All You Need](https://arxiv.org/pdf/1706.03762)

Qwen3 uses the following Transformer block structure:

zh> Qwen3 使用如下的 Transformer 块结构：

```
  input
/ |
| input_layernorm (RMSNorm)
| |
| Qwen3MultiHeadAttention
\ |
  Add (residual)
/ |
| post_attention_layernorm (RMSNorm)
| |
| MLP
\ |
  Add (residual)
  |
output
```

Run the tests for this task with:

zh> 用以下命令运行本任务的测试：

```bash
pdm run test --week 1 --day 5 -- -k task_1
```

## Task 2: Implement `Embedding`

## 任务 2：实现 `Embedding`

```
src/tiny_llm/embedding.py
```

**📚 延伸阅读 / Readings**

- [LLM Embeddings Explained: A Visual and Intuitive Guide](https://huggingface.co/spaces/hesamation/primer-llm-embedding)

The embedding layer maps token IDs (integers) to vectors of length `embedding_dim`. In this task, you will implement
that lookup operation.

zh> 嵌入层把 token ID（整数）映射为长度为 `embedding_dim` 的向量。本任务中，你将实现这一查表操作。

```
Embedding::__call__
weight: vocab_size x embedding_dim
Input: N.. (tokens)
Output: N.. x embedding_dim (vectors)
```

This can be implemented with array indexing.

zh> 这可以用数组索引来实现。

When input and output embeddings are tied, Qwen3 also uses the embedding weight as a linear projection from hidden vectors
back to vocabulary logits.

zh> 当输入与输出嵌入**权重绑定（tied）**时，Qwen3 会直接复用嵌入权重，作为把隐藏向量映射回词表 logits 的线性投影。

```
Embedding::as_linear
weight: vocab_size x embedding_dim
Input: N.. x embedding_dim
Output: N.. x vocab_size
```

Run the tests for this task with:

zh> 用以下命令运行本任务的测试：

```bash
# This task's tests use the 0.6B model and tokenizer.
hf download Qwen/Qwen3-0.6B-MLX-4bit
pdm run test --week 1 --day 5 -- -k task_2
```

## Task 3: Implement `Qwen3ModelWeek1`

## 任务 3：实现 `Qwen3ModelWeek1`

Now that we have built all the Qwen3 components, we can implement `Qwen3ModelWeek1`.

zh> 既然已经实现了 Qwen3 的全部组件，现在可以实现 `Qwen3ModelWeek1` 了。

```
src/tiny_llm/qwen3_week1.py
```

You will not implement the process of reading model parameters from tensor files. Instead, load the model with `mlx_lm`,
then transfer its parameters into our implementation. The `Qwen3ModelWeek1` constructor therefore accepts an MLX model.

zh> 你不需要实现“从张量文件读取模型参数”的过程。取而代之，用 `mlx_lm` 加载模型，再把它的参数搬运进我们的实现中。因此 `Qwen3ModelWeek1` 的构造函数接收一个 MLX 模型对象。

The Qwen3 model has the following layers:

zh> Qwen3 模型包含如下各层：

```
input
| (tokens: N..)
Embedding
| (N.. x hidden_size); note that hidden_size == embedding_dim
Qwen3TransformerBlock
| (N.. x hidden_size)
Qwen3TransformerBlock
| (N.. x hidden_size)
...
|
RMSNorm 
| (N.. x hidden_size)
Embedding.as_linear OR linear (lm_head)
| (N.. x vocab_size)
output
```

Read the number of layers, hidden size, head dimension, and other configuration values from `mlx_model.args`, whose type
is defined by [`ModelArgs`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py). The loaded weights
are available through `mlx_model.model`; use the Qwen3 implementation and model metadata to identify the corresponding
layer names.

zh> 层数、hidden size、head dimension 以及其他配置值，都从 `mlx_model.args` 中读取，其类型由 [`ModelArgs`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py) 定义。加载好的权重可以通过 `mlx_model.model` 访问；请结合 Qwen3 的实现与模型元数据来确定各层对应的名称。

By this point, you have implemented `RMSNorm`. Replace the temporary Day 3 calls to `mx.fast.rms_norm` with
`RMSNorm(head_dim, q_norm, eps=...)` and `RMSNorm(head_dim, k_norm, eps=...)`. They implement the same formula; the built-in
calls existed only to keep the GQA chapter focused on attention.

zh> 到这里，你已经实现了 `RMSNorm`。请把第 3 天临时调用的 `mx.fast.rms_norm` 替换为 `RMSNorm(head_dim, q_norm, eps=...)` 和 `RMSNorm(head_dim, k_norm, eps=...)`。二者实现的是同一个公式；当初使用内置调用，只是为了让 GQA 一章专注在注意力上。

Different Qwen3 model variants map hidden vectors back to vocabulary logits in different ways. Some tie the input and
output embeddings and use `Embedding.as_linear`; others have a separate `lm_head` linear layer. Select the strategy with
`mlx_model.args.tie_word_embeddings`: if it is `True`, use `Embedding.as_linear`; otherwise, load and use `lm_head`.

zh> 不同的 Qwen3 模型变体，把隐藏向量映射回词表 logits 的方式并不相同：有的绑定输入/输出嵌入并使用 `Embedding.as_linear`，有的则带有独立的 `lm_head` 线性层。请依据 `mlx_model.args.tie_word_embeddings` 来选择策略：若为 `True`，使用 `Embedding.as_linear`；否则加载并使用 `lm_head`。

The model takes a sequence of token IDs and returns unnormalized logits for every sequence position. On Day 6, we will
use the final position's logits to select the next token and generate a response.

zh> 该模型接收一串 token ID，并返回每个序列位置上的未归一化 logits（unnormalized logits）。第 6 天，我们将使用最后一个位置的 logits 来选出下一个 token，从而生成回答。

The MLX models used in this course have quantized weights. Dequantize each
linear or embedding layer before loading it into tiny-llm by using the provided
`quantize.dequantize_linear` function, then store the readable Week 1 weight as
BF16. Model activations and layer outputs should remain BF16. A readable
attention or normalization expression may compute in FP32 for stability, but it
must cast its model-facing result back to BF16.

zh> 本课程使用的 MLX 模型权重是量化过的。在把每个线性层或嵌入层加载进 tiny-llm 之前，请先使用已提供的 `quantize.dequantize_linear` 函数对其做反量化，然后把这份“可读的 Week 1 权重”存成 BF16。模型的激活值与各层输出应保持 BF16。为了数值稳定，可读性优先的注意力或归一化表达式可以在 FP32 下计算，但必须把最终对外输出的结果转回 BF16。

Pass `mask="causal"` to every Transformer block. For a one-token sequence the mask has no effect; for longer sequences,
it prevents each position from attending to future tokens.

zh> 向每个 Transformer 块传入 `mask="causal"`。对单 token 序列而言该掩码不起作用；对更长的序列，它能阻止每个位置去 attend 未来的 token。

Run the tests for this task with:

zh> 用以下命令运行本任务的测试：

```bash
# Download each model you want to test. Missing models are skipped.
hf download Qwen/Qwen3-0.6B-MLX-4bit
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
pdm run test --week 1 --day 5 -- -k task_3
```

At the end of the day, you should be able to pass all tests of this day:

zh> 在今天结束时，你应该能通过当天的全部测试：

```bash
pdm run test --week 1 --day 5
```
