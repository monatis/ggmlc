# Operator Reference & Lowering Map

This document provides a comprehensive mapping of all Canonical IR operators, source framework primitives (PyTorch ATen and JAX), and target GGML dialect instructions.

---

## 1. Mathematical & Elementwise Operations

| Canonical Op | PyTorch ATen Targets | JAX Primitives | GGML Lowering | Description |
|---|---|---|---|---|
| `ADD` | `aten.add.Tensor`, `aten.add.Scalar` | `add` | `GGML_OP_ADD` | Elementwise addition $a + b$ |
| `SUB` | `aten.sub.Tensor`, `aten.sub.Scalar` | `sub` | `GGML_OP_SUB` | Elementwise subtraction $a - b$ |
| `MUL` | `aten.mul.Tensor`, `aten.mul.Scalar` | `mul` | `GGML_OP_MUL` | Elementwise multiplication $a \times b$ |
| `DIV` | `aten.div.Tensor`, `aten.div.Scalar` | `div` | `GGML_OP_DIV` | Elementwise division $a / b$ |
| `NEG` | `aten.neg.default` | `neg` | `GGML_OP_NEG` | Elementwise negation $-a$ |
| `POW` | `aten.pow.Tensor_Scalar`, `aten.pow.Tensor_Tensor` | `pow` | `GGML_OP_POW` / `GGML_OP_SQR` | Exponentiation $a^b$ |
| `SQRT` | `aten.sqrt.default` | `sqrt` | `GGML_OP_SQRT` | Square root $\sqrt{a}$ |
| `RSQRT` | `aten.rsqrt.default` | `rsqrt` | `GGML_OP_DIV(1, SQRT(a))` | Reciprocal square root $1/\sqrt{a}$ |
| `SQR` | `aten.square.default` | `square` | `GGML_OP_SQR` | Square $a^2$ |
| `CLAMP` | `aten.clamp.default` | `clamp` | `GGML_OP_CLAMP` | Clamping values to $[\text{min}, \text{max}]$ |

---

## 2. Activation Functions

| Canonical Op | PyTorch ATen Targets | JAX Primitives | GGML Lowering | Description |
|---|---|---|---|---|
| `RELU` | `aten.relu.default` | `relu` | `GGML_OP_RELU` | Rectified linear unit $\max(0, x)$ |
| `GELU` | `aten.gelu.default` | `gelu` | `GGML_OP_GELU` | Gaussian error linear unit |
| `SILU` | `aten.silu.default` | `silu` | `GGML_OP_SILU` | Sigmoid linear unit $x \cdot \sigma(x)$ |
| `SIGMOID` | `aten.sigmoid.default` | `sigmoid` | `GGML_OP_SIGMOID` | Logistic sigmoid $\frac{1}{1 + e^{-x}}$ |
| `TANH` | `aten.tanh.default` | `tanh` | `GGML_OP_TANH` | Hyperbolic tangent $\tanh(x)$ |
| `SOFTMAX` | `aten._softmax.default`, `aten.softmax.default` | `softmax` | `GGML_OP_SOFT_MAX` | Softmax normalized probabilities |
| `SWIGLU` | ATen SwiGLU pattern | `swiglu` | `GGML_OP_GLU` (`ggml_swiglu`) | Swish Gated Linear Unit $x \cdot \text{silu}(g)$ |

---

## 3. Matrix Multiplication & Linear Projections

| Canonical Op | PyTorch ATen Targets | JAX Primitives | GGML Lowering | Description |
|---|---|---|---|---|
| `MATMUL` | `aten.mm.default`, `aten.bmm.default`, `aten.matmul.default` | `dot_general` | `GGML_OP_MUL_MAT` | Matrix multiplication $A \times B$ |
| `LINEAR` | `aten.linear.default`, `aten.addmm.default` | `dot_general` + `add` | `GGML_OP_MUL_MAT` + `bias` | Linear projection $x W^T + b$ |

---

## 4. Tensor Manipulation & Reshaping

| Canonical Op | PyTorch ATen Targets | JAX Primitives | GGML Lowering | Description |
|---|---|---|---|---|
| `TRANSPOSE` | `aten.transpose.int`, `aten.t.default` | `transpose` | `GGML_OP_PERMUTE` | Permutes 2 specified dimensions |
| `PERMUTE` | `aten.permute.default` | `transpose` | `GGML_OP_PERMUTE` | Generalized N-D axis permutation |
| `RESHAPE` / `VIEW` | `aten.view.default`, `aten.reshape.default` | `reshape` | `GGML_OP_RESHAPE` | Reinterprets tensor dimensions |
| `SLICE` | `aten.slice.Tensor` | `slice` | `GGML_OP_VIEW` (`ggml_view_4d`) | Extracts contiguous sub-tensor slice |
| `CONCAT` | `aten.cat.default` | `concatenate` | `GGML_OP_CONCAT` | Concatenates tensors along axis |
| `EXPAND` / `REPEAT`| `aten.expand.default` | `broadcast_in_dim` | `GGML_OP_REPEAT` (`ggml_repeat_4d`)| Broadcasts singleton dimensions |
| `SQUEEZE` | `aten.squeeze.default`, `aten.squeeze.dim` | `squeeze` | `GGML_OP_RESHAPE` | Removes dimensions of size 1 |
| `UNSQUEEZE` | `aten.unsqueeze.default` | `expand_dims` | `GGML_OP_RESHAPE` | Inserts dimension of size 1 |
| `CONTIGUOUS` | `aten.contiguous.default` | — | `GGML_OP_CONT` | Ensures contiguous physical buffer |

---

## 5. Neural Network Layers & Attention

| Canonical Op | PyTorch ATen Targets | JAX Primitives | GGML Lowering | Description |
|---|---|---|---|---|
| `LAYER_NORM` | `aten.layer_norm.default`, `aten.native_layer_norm.default` | `layer_norm` | `GGML_OP_NORM` + affine | Standard Layer Normalization |
| `RMS_NORM` | `aten.rms_norm.default` | `rms_norm` | `GGML_OP_RMS_NORM` | Root Mean Square Normalization |
| `EMBEDDING` | `aten.embedding.default` | `take` | `GGML_OP_GET_ROWS` | Token ID table lookup |
| `ROPE` | Rotary Position Embedding | — | `GGML_OP_ROPE` | Rotary position embeddings on Q/K |
| `SDPA` | `aten.scaled_dot_product_attention.default` | `scaled_dot_product_attention` | `GGML_OP_FLASH_ATTN_EXT` | Fused scaled dot-product attention |
| `MEAN` | `aten.mean.dim` | `reduce_mean` | `GGML_OP_MEAN` | Mean reduction across dimension |
