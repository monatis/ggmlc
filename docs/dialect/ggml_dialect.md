# GGML Dialect Specification & Lowering Semantics

The **GGML Dialect** translates high-level Canonical IR operations into concrete GGML compute graph nodes, memory strides, and kernel calls.

---

## 1. Memory Layout: Row-Major vs. Column-Major

PyTorch and JAX use **row-major (C-order)** memory layouts, where the last dimension changes fastest in contiguous memory.
GGML uses **column-major (Fortran-order)** tensor representation, where `ne[0]` (dimension 0) changes fastest.

### Shape Translation Invariant
For an $R$-dimensional Canonical IR tensor with shape $(d_0, d_1, \dots, d_{R-1})$, the corresponding GGML tensor has extents:
$$ne = [d_{R-1}, d_{R-2}, \dots, d_0, 1, \dots, 1]$$

This ensures that the underlying contiguous 1D memory array in PyTorch maps byte-for-byte to the GGML buffer without requiring physical data transposition.

---

## 2. Permutation & Transposition Mathematics

When lowering a permutation $P$ (where $out\_dim[j] = in\_dim[P[j]]$) from PyTorch to GGML's `ggml_permute(ctx, a, axis0, axis1, axis2, axis3)`:

GGML's `ggml_permute` defines where source axis `ne[i]` is placed in the destination tensor:
$$ne_{dst}[axis_i] = ne_{src}[i]$$

The exact formula to convert from PyTorch permutation $P$ of rank $R$ to GGML's `axis_i` is:
$$\text{dest\_axis}_i = R - 1 - P.\text{index}(R - 1 - i) \quad \text{for } 0 \le i < R$$
$$\text{dest\_axis}_i = i \quad \text{for } R \le i < 4$$

---

## 3. Matrix Multiplication Lowering (`GGML_OP_MUL_MAT`)

In GGML, `ggml_mul_mat(ctx, a, b)` computes:
$$\text{Output} = b \times a^T$$

### Invariant for Parameters (`Linear`)
For PyTorch `Linear(in_features=K, out_features=N)` with weight $W(N, K)$:
- In memory, $W$ has GGML shape $ne = [K, N]$.
- Activation $x$ has GGML shape $ne = [K, M]$.
- Invoking `ggml_mul_mat(W, x)` computes $x(M, K) \times W^T(K, N) = (M, N)$, yielding GGML shape $ne = [N, M]$.
- This is direct and optimal (no transpose required).

### Invariant for General Activation-Activation MatMul (`Q @ K^T` and `Attn @ V`)
When multiplying two activation tensors $A(M, K)$ and $B(K, N)$ where $B$ is not pre-transposed:
- $A$ has $ne = [K, M]$.
- $B$ has $ne = [N, K]$.
- $B$ is transposed via `ggml_transpose(ctx, B)` to $ne = [K, N]$ (tagged with `attrs["transpose_in0"] = 1`).
- `ggml_mul_mat(transpose(B), A)` computes $A(M, K) \times B(K, N) = (M, N)$, strictly matching reference numerical outputs.

---

## 4. Quantized Weight Lowering & Execution

GGML provides native SIMD-accelerated matrix multiplication kernels for quantized weights:

| GGML Type | Value | Layout | Contiguous Constraint |
| :--- | :--- | :--- | :--- |
| `GGML_TYPE_F32` | `0` | 4 bytes/elem | Standard |
| `GGML_TYPE_Q4_0` | `2` | 18 bytes / 32 elements | **Must be strictly contiguous row blocks** |
| `GGML_TYPE_Q8_0` | `8` | 34 bytes / 32 elements | **Must be strictly contiguous row blocks** |

### Critical Quantization Rules:
1. **Pre-Quantization Transposition**: Any Hugging Face `Conv1D` or non-standard parameter tensor is physically transposed in memory **before** quantization, so that $ne = [K, N]$ stores contiguous rows of $K$ elements in quantized blocks.
2. **No Graph-Level Non-Contiguous Ops on Quantized Tensors**: `ggml_transpose`, `ggml_permute`, and `ggml_cont` cannot be invoked on quantized weight tensors. The runtime dispatches directly to quantized GEMM without auxiliary graph ops.

---

## 5. Grouped Query Attention (GQA) Lowering
In models with Grouped Query Attention (e.g. Qwen2.5, LLaMA-3), the number of KV heads $N_{kv}$ is smaller than query heads $N_q$.
- In PyTorch: `repeat_interleave` duplicates each KV head $N_q / N_{kv}$ times contiguously.
- In GGML: `ggml_repeat` tiles the entire tensor.
- Lowering Strategy: GQA KV tensors are lowered via a 4D view $(D, S, \text{num\_groups}, N_{kv})$, repeated along the group dimension, and flattened back to $(D, S, N_q, B)$ before attention computation.

---

## 6. Vision & Convolutional Operators

- **2D Convolution (`GGML_OP_CONV_2D`)**:
  - Direct kernel dispatch: `ggml_conv_2d(ctx, weight, x, s0, s1, p0, p1, d0, d1)`.
  - In GGML column-major layout:
    - Weight tensor: $ne = [K_w, K_h, C_{in}, C_{out}]$.
    - Input tensor: $ne = [W, H, C_{in}, B]$.
    - Output tensor: $ne = [W_{out}, H_{out}, C_{out}, B]$.
- **2D Pooling (`GGML_OP_POOL_2D`)**:
  - Dispatches `ggml_pool_2d(ctx, x, type, k0, k1, s0, s1, p0, p1)`.
  - Supports `GGML_OP_POOL_MAX` and `GGML_OP_POOL_AVG`.
  - Adaptive Average Pooling computes dynamic kernel $k = (W, H)$ and stride $s = (W, H)$ to reduce feature maps to $1 \times 1$ spatial grids.

---

## 7. Normalization & Activations

- **LayerNorm (`GGML_OP_NORM`)**: Computes mean and variance over dimension 0 ($ne[0]$). Broadcast scale & shift applied via `ggml_repeat` + `ggml_mul` + `ggml_add`.
- **RMSNorm (`GGML_OP_RMS_NORM`)**: Computes root-mean-square normalization over dimension 0 ($ne[0]$).
- **Softmax (`GGML_OP_SOFT_MAX`)**: Computes normalized exponentials along dimension 0.
- **RoPE (`GGML_OP_ROPE`)**: Applies rotary position embeddings directly to $Q$ and $K$ heads.
- **SwiGLU (`GGML_OP_GLU`)**: Native implementation of the SwiGLU activation $x \odot \text{silu}(g)$.
