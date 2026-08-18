# Canonical IR Specification

The `ggmlc` **Canonical IR** is a framework-independent intermediate representation designed to model neural networks as semantic tensor programs.

---

## 1. Core Principles

1. **A neural network is a functional semantic tensor program.**
2. **Framework-agnostic**: The IR represents pure mathematical and layout operations, independent of whether the source model originated from PyTorch, JAX, or ONNX.
3. **Symbolic Shapes as First-Class Values**: Dimensions can be dynamic symbolic expressions evaluated at runtime.
4. **Explicit Storage & Lifetimes**: Every tensor carries an explicit `StorageClass` defining its memory semantics.

---

## 2. Type System (`DType`)

| Enum | Description | Typical Size |
|---|---|---|
| `DType.F32` | 32-bit single-precision float | 4 bytes |
| `DType.F16` | 16-bit half-precision float | 2 bytes |
| `DType.BF16` | 16-bit brain float | 2 bytes |
| `DType.I32` | 32-bit signed integer | 4 bytes |
| `DType.I16` | 16-bit signed integer | 2 bytes |
| `DType.I8` | 8-bit signed integer | 1 byte |
| `DType.I64` | 64-bit signed integer | 8 bytes |
| `DType.BOOL` | Boolean flag | 1 byte |
| `DType.Q8_0`, `Q4_0`, `Q4_K` | Quantized block formats | Variable |

---

## 3. Shape & Symbolic Dimensions (`Dim`, `Shape`)

The shape system models arbitrary ranks with both static integers and symbolic dimension expressions:

```python
class Dim(ABC):
    def evaluate(self, env: dict[str, int]) -> int: ...
```

### Dimension Expression Hierarchy
- `StaticDim(value: int)`: Constant integer dimension extent (e.g. `StaticDim(768)`).
- `SymbolDim(name: str)`: Named symbolic parameter bound at runtime (e.g. `SymbolDim("batch")`, `SymbolDim("seq")`).
- `AddDim(left, right)`: Sum of two dimension expressions ($d_1 + d_2$).
- `SubDim(left, right)`: Difference of two dimension expressions ($d_1 - d_2$).
- `MulDim(left, right)`: Product of two dimension expressions ($d_1 \times d_2$).
- `FloorDivDim(left, right)`: Integer floor division ($\lfloor d_1 / d_2 \rfloor$).
- `CeilDivDim(left, right)`: Integer ceiling division ($\lceil d_1 / d_2 \rceil$).

---

## 4. Tensor Storage Classes (`StorageClass`)

Each tensor in the graph is tagged with its role in the computation lifecycle:

| StorageClass | Description | Lifetime |
|---|---|---|
| `INPUT` | User-provided model input tensor | Ephemeral (bound per inference run) |
| `PARAMETER` | Trainable model weight or bias | Static (embedded in compiled artifact) |
| `CONSTANT` | Constant scalar or static tensor | Static (embedded in compiled artifact) |
| `ACTIVATION` | Intermediate computed activation buffer | Ephemeral (managed by runtime memory pool) |
| `STATE` | Persistent mutable buffer (e.g. KV Cache) | Persistent across consecutive inference runs |
| `OUTPUT` | Final graph output tensor | Read out after execution |

---

## 5. Canonical Operator Set (`OpCode`)

### Elementwise Arithmetic & Math
- `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `POW`, `SQR`, `SQRT`, `RSQRT`, `EXP`, `LOG`, `ABS`, `CLAMP`

### Activations
- `RELU`, `GELU`, `SILU`, `SIGMOID`, `TANH`, `SOFTMAX`, `SWIGLU`

### Linear Algebra & Matrix Multiplication
- `MATMUL`: General matrix product ($A \times B$).
- `LINEAR`: Linear layer projection ($x \times W^T + b$).

### Tensor Manipulation & Structural Ops
- `TRANSPOSE`: 2D/N-D axis swap with dimension normalization.
- `PERMUTE`: Arbitrary N-dimensional axis rearrangement.
- `RESHAPE`, `VIEW`: Logical shape reinterpretation.
- `SLICE`: Sub-tensor slicing along specified dimension with start, end, step.
- `CONCAT`: Concatenation of multiple tensors along a specified axis.
- `EXPAND`, `REPEAT`: Dimension broadcasting.
- `SQUEEZE`, `UNSQUEEZE`: Dimension removal or insertion.
- `CONTIGUOUS`: Guarantees contiguous memory buffer layout.

### Normalization & High-Level Primitives
- `LAYER_NORM`: Standard Layer Normalization with affine weight and bias.
- `RMS_NORM`: Root Mean Square Layer Normalization.
- `EMBEDDING`: Token table row lookup (`ggml_get_rows`).
- `ROPE`: Rotary Position Embedding.
- `SDPA`: Scaled Dot-Product Attention.
- `MEAN`, `SUM`: Reduction operations along specified axes.
