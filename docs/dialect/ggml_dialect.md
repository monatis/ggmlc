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

## 4. Normalization Operators

- **LayerNorm (`GGML_OP_NORM`)**:
  Computes mean and variance over the inner dimension:
  $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
  Affine scale (weight) and shift (bias) are broadcast and applied via `ggml_repeat` + `ggml_mul` + `ggml_add`.
- **RMSNorm (`GGML_OP_RMS_NORM`)**:
  Computes root-mean-square normalization over the inner dimension:
  $$\hat{x} = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$$

---

## 5. Attention & RoPE

- **`GGML_OP_SOFT_MAX`**: Applied along dimension 0 ($ne[0]$).
- **`GGML_OP_FLASH_ATTN_EXT`**: Native fused scaled dot-product attention kernel with optional causal masking.
- **`GGML_OP_ROPE`**: Applies rotary position embeddings directly to $Q$ and $K$ heads.
- **`GGML_OP_GLU` (`ggml_swiglu`)**: Native implementation of the SwiGLU activation $x \odot \text{silu}(g)$.
