# Day 4: RMSNorm and the Multilayer Perceptron

# 第 4 天：RMSNorm 与多层感知机

On Day 4, we will implement two important components of the Qwen3 Transformer architecture: RMSNorm and the multilayer
perceptron (MLP), also known as the feed-forward network. RMSNorm is a normalization technique with less computational
overhead than traditional layer normalization. The MLP applies nonlinear transformations after the attention block.

zh> 第 4 天，我们将实现 Qwen3 Transformer 架构中的两个重要组件：RMSNorm 与多层感知机（MLP，也称前馈网络）。RMSNorm 是一种归一化方法，计算开销比传统的层归一化（LayerNorm）更低。MLP 则在注意力块之后施加非线性变换。

## Task 1: Implement `RMSNorm`

## 任务 1：实现 `RMSNorm`

In this task, we will implement the `RMSNorm` layer.

zh> 本任务将实现 `RMSNorm` 层。

```
src/tiny_llm/layer_norm.py
```

Day 3 used `mx.fast.rms_norm` directly so that the GQA chapter could stay focused on attention. This task implements the
same normalization rule as a reusable layer. From this point on, the Transformer block, final model normalization, and
Q/K normalization path can use your `RMSNorm` implementation.

zh> 第 3 天直接调用了 `mx.fast.rms_norm`，是为了让 GQA 一章专注在注意力本身。本任务则把同一套归一化规则实现成一个可复用的层。从现在起，Transformer 块、模型最后的归一化，以及 Q/K 归一化路径，都可以改用你自己的 `RMSNorm` 实现。

**📚 延伸阅读 / Readings**

* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
* [Qwen3 layers implementation in mlx-lm (includes RMSNorm)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py) - See `RMSNorm`.

RMSNorm is defined as:

zh> RMSNorm 的定义如下：

$$
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \text{weight}
$$

where:

zh> 其中：

- `x` is the input tensor.
- `weight` is a learned scaling parameter.
- `epsilon` (`eps`) is a small constant, such as `1e-5` or `1e-6`, added for numerical stability.
- `mean(x^2)` is the mean of the squared elements along the final dimension.

zh> - `x` 是输入张量；
zh> - `weight` 是可学习的缩放参数；
zh> - `epsilon`（`eps`）是为了数值稳定而加入的极小常数，例如 `1e-5` 或 `1e-6`；
zh> - `mean(x^2)` 是沿最后一个维度求得的平方元素均值。

Apply normalization independently to each feature vector along the input's final dimension. Cast the input to `float32`
for the normalization calculation, including the mean, to preserve precision when the original values use `float16` or
`bfloat16`. Cast the normalized value back to the input dtype before applying `weight`. This matches the low-precision
path used by MLX's fast RMSNorm kernels: normalization statistics are accumulated in `float32`, while the final scaling
happens in the model dtype.

zh> 请沿着输入的最后一个维度，对每个特征向量独立地做归一化。做归一化计算（包括求均值）时，先把输入转成 `float32`，以在原始值为 `float16` 或 `bfloat16` 时保住精度；在乘上 `weight` 之前，再把归一化后的值转回输入原始 dtype。这与 MLX 快速 RMSNorm 内核的低精度路径一致：归一化统计量在 `float32` 中累加，而最后的缩放仍在模型 dtype 下进行。

```
D is the embedding dimension.

x: N.. x D
weight: D
output: N.. x D
```

You can test your implementation by running:

zh> 可以运行以下命令测试你的实现：

```bash
pdm run test --week 1 --day 4 -- -k task_1
```

## Task 2: Implement the MLP Block

## 任务 2：实现 MLP 块

In this task, we will implement the MLP block named `Qwen3MLP`.

zh> 本任务将实现名为 `Qwen3MLP` 的 MLP 块。

```
src/tiny_llm/qwen3_week1.py
```

The original Transformer uses a simple position-wise feed-forward network (FFN) in each block. It consists of two linear
transformations with a ReLU activation between them.

zh> 最初的 Transformer 在每个块中使用简单的逐位置前馈网络（FFN），它由两个线性变换及其之间的 ReLU 激活组成。

Modern Transformer architectures, including Qwen3, often use more advanced FFN variants. Qwen3 uses SwiGLU, a gated linear
unit (GLU) variant.

zh> 包括 Qwen3 在内的现代 Transformer 架构通常使用更先进的 FFN 变体。Qwen3 使用 SwiGLU，一种门控线性单元（GLU）变体。

A plain FFN can be abstracted as:

zh> 一个普通 FFN 可以抽象为：

```plain
h = activation(W_up(x))
out = W_down(h)
```

A GLU keeps the same expand-then-project-back shape but adds another projection that gates the intermediate features before
`W_down`. This gives the MLP a learned, input-dependent way to control which intermediate channels matter, rather than
applying an activation only to the features produced by `W_up`.

zh> GLU 保留了同样的“先升维、再投影回去”的结构，但在 `W_down` 之前额外加入一个投影，用来对中间特征做门控。这让 MLP 拥有一种可学习、依赖输入的方式来控制哪些中间通道更重要，而不仅仅是对 `W_up` 产生的特征施加一个激活函数。

SwiGLU is the GLU variant used by Qwen3:

zh> SwiGLU 即 Qwen3 采用的 GLU 变体：

```plain
u = W_up(x)
g = SiLU(W_gate(x))
out = W_down(g * u)
```

**📚 延伸阅读 / Readings**

- [Attention is All You Need (Transformer Paper, Section 3.3 "Position-wise Feed-Forward Networks")](https://arxiv.org/abs/1706.03762)
- [GLU paper: Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083)
- [SiLU (Swish) activation function](https://arxiv.org/pdf/1710.05941)
- [SwiGLU paper: GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)
- [PyTorch SiLU documentation](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [Qwen3 layers implementation in mlx-lm (includes MLP)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)

SwiGLU combines a GLU with the SiLU (sigmoid linear unit) activation function:

zh> SwiGLU 把 GLU 与 SiLU（sigmoid linear unit）激活函数结合起来：

- A GLU gates one linear projection of the input with another, using element-wise multiplication to control which features
  pass through.
- SiLU is a smooth, non-monotonic activation function. Unlike ReLU, it has no zero-gradient region across all negative
  inputs and can produce nonzero outputs for negative values.

zh> - GLU 用输入的另一个线性投影去门控其中一个线性投影，通过逐元素乘法控制哪些特征得以通过；
zh> - SiLU 是平滑且非单调的激活函数。与 ReLU 不同，它在所有负输入上并不存在梯度恒为零的区域，并且对负值也能产生非零输出。

First, implement `silu` in `basics.py`. It takes a tensor of shape `N.. x I` and returns a tensor with the same shape:

zh> 首先，在 `basics.py` 中实现 `silu`。它接收形状为 `N.. x I` 的张量，并返回同形状张量：

$$
\text{SiLU}(x) = x * \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}
$$

Compute the sigmoid part in a numerically stable way:

zh> 请以数值稳定的方式计算 sigmoid 部分：

```text
if x >= 0:
    sigmoid(x) = 1 / (1 + exp(-x))
else:
    sigmoid(x) = exp(x) / (1 + exp(x))
```

The negative branch is algebraically equivalent to the direct sigmoid formula, but it prevents `exp(-x)` from becoming
`exp(large positive)` when `x` is a large negative value. In vector code, first compute `z = exp(-abs(x))`. Use
`z / (1 + z)` for negative inputs and `1 / (1 + z)` otherwise. Do not rewrite the negative branch as
`1 - 1 / (1 + z)`: in low precision, the fraction can round to `1`, and the subtraction then incorrectly produces zero.

zh> 负半轴分支在代数上与 sigmoid 的直接公式等价，但它避免了当 `x` 为很大负数时 `exp(-x)` 变成 `exp(很大的正数)` 而溢出。在向量化代码中，先计算 `z = exp(-abs(x))`：负输入用 `z / (1 + z)`，其余用 `1 / (1 + z)`。不要把负半轴分支改写成 `1 - 1 / (1 + z)`：在低精度下这个分式可能舍入到 `1`，随后的减法就会错误地得到 0。

Then implement `Qwen3MLP`. Qwen3's MLP contains:

zh> 接着实现 `Qwen3MLP`。Qwen3 的 MLP 包含：

- A gate projection ($W_{gate}$)
- An up projection ($W_{up}$)
- SiLU applied to the gate projection's output
- An element-wise product of the activated gate output and the up-projection output
- A final down projection ($W_{down}$)

zh> - 门控投影（$W_{gate}$）
zh> - 升维投影（$W_{up}$）
zh> - 对门控投影输出施加 SiLU
zh> - 将激活后的门控输出与升维投影输出逐元素相乘
zh> - 最后的降维投影（$W_{down}$）

This can be expressed as:

zh> 可以表示为：

$$
\text{MLP}(x) = W_{down}(\text{SiLU}(W_{gate}(x)) \odot W_{up}(x))
$$

where $\odot$ denotes element-wise multiplication. Qwen3's MLP projections do not use biases.

zh> 其中 $\odot$ 表示逐元素相乘。Qwen3 的 MLP 投影均不使用偏置。

```
N.. is zero or more dimensions for batches
E is hidden_size (embedding dimension of the model)
I is intermediate_size (dimension of the hidden layer in MLP)
L is the sequence length

input: N.. x L x E
w_gate: I x E
w_up: I x E
w_down: E x I
output: N.. x L x E
```

You can test your implementation by running:

zh> 可以运行以下命令测试你的实现：

```bash
pdm run test --week 1 --day 4 -- -k task_2
```

At the end of the day, you should be able to pass all tests of this day:

zh> 在今天结束时，你应该能通过当天的全部测试：

```bash
pdm run test --week 1 --day 4
```
