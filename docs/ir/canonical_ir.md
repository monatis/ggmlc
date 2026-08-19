# Canonical IR Specification

The `ggmlc` **Canonical IR** is a framework-independent intermediate representation designed to model neural networks as semantic tensor programs.

---

## 1. Core Principles

1. **A neural network is a functional semantic tensor program.**
2. **Framework-agnostic**: The IR represents pure mathematical and layout operations, independent of whether the source model originated from PyTorch, JAX, or ONNX.
3. **Symbolic Shapes as First-Class Values**: Dimensions can be dynamic symbolic expressions evaluated at runtime.
4. **Explicit Storage & Lifetimes**: Every tensor carries an explicit `StorageClass` defining its memory semantics.
5. **Optimization-Ready**: Designed for direct compile-time transformation passes (Constant Folding, DCE, Pattern Fusion).

---

## 2. Type System (`DType`)

| Enum | Description | Bytes / Block | `is_quantized` |
|---|---|---|---|
| `DType.F32` | 32-bit single-precision float | 4 bytes | `False` |
| `DType.F16` | 16-bit half-precision float | 2 bytes | `False` |
| `DType.BF16` | 16-bit brain float | 2 bytes | `False` |
| `DType.I32` | 32-bit signed integer | 4 bytes | `False` |
| `DType.I16` | 16-bit signed integer | 2 bytes | `False` |
| `DType.I8` | 8-bit signed integer | 1 byte | `False` |
| `DType.I64` | 64-bit signed integer | 8 bytes | `False` |
| `DType.BOOL` | Boolean flag | 1 byte | `False` |
| `DType.Q8_0` | GGML 8-bit block quantization (32 quants + fp16 scale) | 34 bytes / 32 floats | `True` |
| `DType.Q4_0` | GGML 4-bit block quantization (32 nibbles + fp16 scale) | 18 bytes / 32 floats | `True` |
| `DType.Q4_K` | GGML k-quants 4-bit super-block quantization | Variable | `True` |

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

`Shape.is_dynamic` returns `True` if any dimension contains non-static symbolic expressions.

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

## 5. Canonical Transformation Passes (`ggmlc.transforms`)

Canonical IR graphs are optimized prior to target lowering via `PassManager`:
- **`ConstantFoldingPass`**: Pre-evaluates deterministic constant subgraphs at compile time.
- **`OperatorFusionPass`**: Fuses Conv2D+ReLU, Linear+Bias, and SwiGLU composite patterns.
- **`DeadCodeEliminationPass`**: Traverses backward from outputs and states to eliminate unused nodes.
- **`RedundantCastPruner`**: Removes no-op transpositions and identical dtype casts.

---

## 6. Canonical Operator Set (`OpCode`)

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

### Spatial Convolutions & Pooling
- `CONV2D`: 2D Spatial Convolution with kernel, stride, and padding attributes.
- `MAX_POOL2D`, `AVG_POOL2D`: 2D Pooling with adaptive or explicit kernel parameters.

### Normalization & High-Level Primitives
- `LAYER_NORM`: Standard Layer Normalization with affine weight and bias.
- `RMS_NORM`: Root Mean Square Layer Normalization.
- `EMBEDDING`: Token table row lookup (`ggml_get_rows`).
- `ROPE`: Rotary Position Embedding.
- `SDPA`: Scaled Dot-Product Attention with optional causal or attention mask.
- `MEAN`, `SUM`: Reduction operations along specified axes.
