# Day 3: Grouped Query Attention (GQA)

On Day 3, we will implement grouped-query attention (GQA). Qwen3 uses GQA to reduce the computational and memory costs
of the key (K) and value (V) projections. In multi-head attention (MHA), every query (Q) head has a corresponding K and
V head. With GQA, groups of Q heads share K and V heads. Multi-query attention (MQA) is the special case in which every
Q head shares a single K/V head pair.

zh> 第 3 天，我们将实现分组查询注意力（Grouped-Query Attention，GQA）。Qwen3 使用 GQA 来降低键（K）与值（V）投影的计算与内存开销。在多头注意力（MHA）中，每个查询（Q）头都有与之对应的 K 头和 V 头；而在 GQA 中，若干个 Q 头组成一组，共享同一组 K/V 头。多查询注意力（MQA）是 GQA 的特例：所有 Q 头共享唯一的一对 K/V 头。

**延伸阅读 / Readings**

- [GQA paper](https://arxiv.org/abs/2305.13245)
- [Qwen3 layers in mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [PyTorch scaled dot-product attention with `enable_gqa=True`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [`torchtune.modules.MultiHeadAttention`](https://pytorch.org/torchtune/0.3/generated/torchtune.modules.MultiHeadAttention.html)

## Task 1: Implement `scaled_dot_product_attention_grouped`

## 任务 1：实现 `scaled_dot_product_attention_grouped`

You will need to modify the following file:

zh> 你需要修改以下文件：

```
src/tiny_llm/attention.py
```

In this task, we will implement grouped scaled dot-product attention, which forms the core of GQA.

zh> 本任务将实现“分组缩放点积注意力”，它是 GQA 的核心。

Implement `scaled_dot_product_attention_grouped` in `src/tiny_llm/attention.py`. It is similar to standard scaled dot-product
attention, but it supports a number of query heads that is a multiple of the number of key/value heads.

zh> 请在 `src/tiny_llm/attention.py` 中实现 `scaled_dot_product_attention_grouped`。它与标准的缩放点积注意力类似，区别在于：查询头的数量可以是键/值头数量的整数倍。

The main process is the same as standard scaled dot-product attention. The difference is that K and V heads are shared
across multiple Q heads. Instead of `H_q` separate K and V heads, there are `H` K and V heads, each shared by
`n_repeats = H_q // H` query heads.

zh> 整体流程与标准缩放点积注意力一致，区别在于 K/V 头被多个 Q 头共享：不存在 `H_q` 个彼此独立的 K/V 头，而是只有 `H` 个 K/V 头，每个被 `n_repeats = H_q // H` 个查询头共享。

Reshape `query`, `key`, and `value` so that K and V can be broadcast to the query heads in their respective groups during
the matrix multiplications.

zh> 请调整 `query`、`key`、`value` 的形状，使 K 和 V 能在矩阵乘法中广播到各自分组内的所有查询头上。

- Separate the `H` and `n_repeats` dimensions in `query`.
- Add a dimension of size 1 for `n_repeats` in `key` and `value` so that they broadcast across each group.

zh> - 在 `query` 中把 `H` 与 `n_repeats` 两个维度拆开；
zh> - 在 `key` 和 `value` 中插入一个大小为 1 的 `n_repeats` 维度，使其能在每个分组内广播。

Then perform scaled dot-product attention: matrix multiplication, scaling, optional masking, softmax, and a final matrix
multiplication. Broadcasting handles the head sharing without materializing repeated K and V tensors.

zh> 随后执行缩放点积注意力的标准步骤：矩阵乘法、缩放、可选的掩码、softmax，以及最后一次矩阵乘法。借助广播即可实现头共享，无需真正把 K、V 复制扩展成重复的大张量。

Using broadcasting instead of repeating K and V is more efficient because it avoids creating copies of the same data.

zh> 相比“重复复制”K 和 V，使用广播更高效，因为它避免了创建同一份数据的多个副本。

Finally, reshape the result to the expected output shape.

zh> 最后，把结果 reshape 成期望的输出形状。

```
N.. is zero or more dimensions for batches
H_q is the number of query heads
H is the number of key/value heads (H_q must be divisible by H)
L is the query sequence length
S is the key/value sequence length
D is the head dimension

query: N.. x H_q x L x D
key: N.. x H x S x D
value: N.. x H x S x D
mask: N.. x H_q x L x S
output: N.. x H_q x L x D
```

In addition to grouped heads, this function supports different query and key/value sequence lengths: Q uses length `L`,
while K and V use length `S`.

zh> 除了分组头之外，该函数还支持查询与键/值使用不同的序列长度：Q 的长度为 `L`，而 K、V 的长度为 `S`。

You can test your implementation by running the following command:

zh> 可以运行以下命令测试你的实现：

```bash
pdm run test --week 1 --day 3 -- -k task_1
```

## Task 2: Causal Masking

## 任务 2：因果掩码

**延伸阅读 / Readings**

- [Writing an LLM from scratch, part 9 -- causal attention](https://www.gilesthomas.com/2025/03/llm-from-scratch-9-causal-attention)

In this task, we will add causal masking to grouped attention.

zh> 本任务将为分组注意力加上因果掩码（causal masking）。

Causal masking prevents attention from reading future tokens. When `mask` is set to the string `"causal"`, apply a causal
mask.

zh> 因果掩码用于阻止注意力读取未来的 token。当 `mask` 被设为字符串 `"causal"` 时，就应用因果掩码。

The additive causal mask has shape `(L, S)`, where `L` is the query sequence length and `S` is the key/value sequence length.
Allowed positions contain 0, and masked positions contain `-inf`. When `S` is greater than `L`, shift the diagonal by
`S - L` so that the queries correspond to the final `L` positions in the key/value sequence. For example, if `L = 3`
and `S = 5`, the mask is:

zh> 这个加性（additive）因果掩码的形状为 `(L, S)`，其中 `L` 是查询序列长度，`S` 是键/值序列长度。允许 attend 的位置取值为 0，被屏蔽的位置取值为 `-inf`。当 `S` 大于 `L` 时，需要把对角线偏移 `S - L`，使查询对应键/值序列中最后的 `L` 个位置。例如当 `L = 3`、`S = 5` 时，掩码为：

```
0   0   0   -inf -inf
0   0   0   0    -inf
0   0   0   0    0
```

Implement `causal_mask` in `src/tiny_llm/attention.py`, then use it in `scaled_dot_product_attention_grouped`. Note that
our shifted diagonal for `L != S` differs from the default behavior of some attention APIs.

zh> 请在 `src/tiny_llm/attention.py` 中实现 `causal_mask`，并在 `scaled_dot_product_attention_grouped` 中使用它。注意：我们在 `L != S` 时采用的对角线偏移方式，与某些注意力 API 的默认行为不同。

You can test your implementation by running the following command:

zh> 可以运行以下命令测试你的实现：

```bash
pdm run test --week 1 --day 3 -- -k task_2
```

## Task 3: Qwen3 Grouped Query Attention

## 任务 3：Qwen3 的分组查询注意力

In this task, we will implement Qwen3's grouped-query attention. Modify the following file:

zh> 本任务将实现 Qwen3 的分组查询注意力。请修改以下文件：

```
src/tiny_llm/qwen3_week1.py
```

`Qwen3MultiHeadAttention` implements attention for Qwen3. Follow this pseudocode:

zh> `Qwen3MultiHeadAttention` 实现了 Qwen3 的注意力。请按如下伪代码实现：

```
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(0, L))
k = rope(k, offset=slice(0, L))
(transpose as needed)
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, H_q, L, D  # use float32
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

Qwen3 attention has no Q/K/V projection biases, and it applies RMSNorm to each Q and K head before RoPE. We will implement
the reusable `RMSNorm` layer on Day 4, so call `mx.fast.rms_norm` directly for `q_norm` and `k_norm` today. Use
non-traditional RoPE.

zh> Qwen3 的注意力中 Q/K/V 投影都不带偏置，并且在 RoPE 之前对每个 Q 头和 K 头分别做 RMSNorm。可复用的 `RMSNorm` 层会在第 4 天实现，因此今天请直接调用 `mx.fast.rms_norm` 来完成 `q_norm` 和 `k_norm`。RoPE 使用非传统（non-traditional）布局。

You can test your implementation by running the following command:

zh> 可以运行以下命令测试你的实现：

```bash
pdm run test --week 1 --day 3 -- -k task_3
```

At the end of the day, you should be able to pass all tests of this day:

zh> 在今天结束时，你应该能通过当天的全部测试：

```bash
pdm run test --week 1 --day 3
```
