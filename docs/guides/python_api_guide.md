# ggmlc Python High-Performance API & User Guide

`ggmlc` provides a modern, unified, framework-agnostic Python API for compiling, optimizing, quantizing, and executing neural network models from **PyTorch**, **JAX**, and **Flax** directly onto GGML and CPU hardware backends.

---

## 1. Quickstart: Compiling & Running Models in 3 Lines

```python
import ggmlc
import torch
import torchvision.models as models

# 1. Take any standard PyTorch or Flax model
model = models.resnet18(weights=None).eval()
example_input = torch.randn(1, 3, 224, 224)

# 2. Compile directly into self-contained standard GGUF binary bytes
gguf_bytes = ggmlc.compile(
    model=model,
    sample_inputs=(example_input,),
    model_name="resnet18",
)

# 3. Load into high-performance native C++ nanobind runtime
runner = ggmlc.load(gguf_bytes, n_threads=4)

# 4. Evaluate with zero-copy numpy buffers
output = runner(example_input.numpy())
print("Output shape:", output.shape)
```

---

## 2. Core Python APIs

### `ggmlc.compile`
Compiles a PyTorch model, JAX callable, or Flax module into standardized GGUF v3 binary format.

```python
def compile(
    model: Any,
    sample_inputs: tuple[Any, ...],
    dynamic_shapes: dict[str, Any] | None = None,
    optimization_level: int = 1,
    quantization_type: str | None = None,
    output_path: str | Path | None = None,
    model_name: str = "model",
) -> bytes:
```

- **`model`**: A `torch.nn.Module`, `torch.export.ExportedProgram`, JAX pure function (`Callable`), or Flax `nn.Module`.
- **`sample_inputs`**: Tuple of sample tensor/array inputs used for tracing and shape inference.
- **`dynamic_shapes`**: Optional specification for dynamic dimensions (e.g. `torch.export.Dim`).
- **`optimization_level`**: Standard optimization pipeline level (dead code elimination, redundant cast pruning, constant folding).
- **`quantization_type`**: Optional quantization format: `"q8_0"`, `"q4_0"`, `"int8"`, `"int4"`.
- **`output_path`**: Optional file path to save the `.gguf` binary on disk.
- **`model_name`**: Name of the model embedded in metadata.

---

### `ggmlc.load`
Loads a compiled model from a `.gguf` file path or in-memory `bytes` into a `ModelRunner`.

```python
runner = ggmlc.load("model.gguf", n_threads=4)
# Or from memory bytes:
runner = ggmlc.load(gguf_bytes, n_threads=4)
```

### `ModelRunner` Invocation
```python
# Positional call
out = runner(x_numpy)

# Named tensor call
out = runner(input_ids=ids_numpy)

# Dynamic shape symbol override
out = runner(x_numpy, symbols={"seq": 64})
```

---

### `ggmlc.codegen`
Generates an independent, standalone C++ model library and executable from a compiled model:

```python
artifacts = ggmlc.codegen(
    model=model,
    sample_inputs=(sample_x,),
    output_dir="./build_cpp_model",
    model_name="my_model",
)
```

This generates:
- `model.h`: Self-contained C++ header defining model tensor allocations and computation graph.
- `ggmlc_main.cpp`: Standalone CLI driver for running inference and benchmarks.
- `CMakeLists.txt`: Build script configured to compile natively against GGML.

---

## 3. Full JAX & Flax Support

`ggmlc` natively ingests JAX expressions (`jaxpr`) and Flax linen modules:

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import ggmlc

class FlaxClassifier(nn.Module):
    hidden_dim: int = 128
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        return nn.Dense(self.num_classes)(x)

model = FlaxClassifier()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32)))

def forward(x):
    return model.apply(params, x)

# Compile Flax forward function
gguf_bytes = ggmlc.compile(
    model=forward,
    sample_inputs=(jnp.ones((1, 32)),),
    model_name="flax_classifier",
)

# Run inference
runner = ggmlc.load(gguf_bytes)
out = runner(jnp.ones((1, 32)))
```

---

## 4. Native Multi-Platform Compilation

### Building on Windows (MSVC)
```powershell
cmake -B build-win
cmake --build build-win --target _runtime --config Release
```

### Building on Linux / WSL (GCC / Clang)
```bash
cmake -B build
cmake --build build --target _runtime -j$(nproc)
```
