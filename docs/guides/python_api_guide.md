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

# 2. Compile directly into a standard GGUF binary file (streamed directly to disk)
model_path = ggmlc.compile(
    model=model,
    sample_inputs=(example_input,),
    output="resnet18.gguf",
    model_name="resnet18",
)

# 3. Load into high-performance native C++ nanobind runtime
runner = ggmlc.load(model_path, n_threads=4)

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
    sample_inputs: tuple[Any, ...] | list[Any] | None = None,
    output: str | Path | None = None,
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    return_runner: bool = False,
    **kwargs: Any,
) -> Path | bytes | ModelRunner:
```

- **`model`**: A `torch.nn.Module`, `torch.export.ExportedProgram`, JAX pure function (`Callable`), or Flax `nn.Module`.
- **`sample_inputs`**: Tuple of sample tensor/array inputs used for tracing and shape inference.
- **`output`**: Optional file path to stream the `.gguf` binary directly to disk (prevents intermediate RAM allocation).
- **`dynamic_shapes`**: Optional specification for dynamic dimensions (e.g. `torch.export.Dim`).
- **`model_name`**: Name of the model embedded in graph and GGUF metadata.
- **`enable_optimizations`**: If `True`, runs standard optimization passes (constant folding, dead-code elimination, redundant cast pruning).
- **`enable_fusion`**: If `True`, lowers composite subgraphs into high-performance fused ops.
- **`quantize`**: Optional parameter quantization format: `"q8_0"`, `"q4_0"`, `DType.Q8_0`, `DType.Q4_0`.
- **`return_runner`**: If `True`, automatically loads and returns an instantiated `ModelRunner`.

### `ggmlc.compile_to_bytes`
Helper function for compiling and returning raw in-memory GGUF bytes when needed:
```python
gguf_bytes = ggmlc.compile_to_bytes(model, sample_inputs=(x,), model_name="my_model")
```

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
- `my_model.h`: Self-contained C++ header defining model tensor allocations and computation graph.
- `ggmlc_main.cpp`: Standalone CLI driver for running inference and benchmarks.
- `CMakeLists.txt`: Build script configured to compile natively against GGML.

---

### `ggmlc.visualize`
Visualizes Canonical IR or GGML dialect graphs and exports them to PNG, SVG, interactive HTML, or Mermaid markdown:

```python
png_path = ggmlc.visualize(graph, output_path="model.png", format="png")   # Render directly to PNG
svg_path = ggmlc.visualize(graph, output_path="model.svg", format="svg")   # Vector SVG
html_path = ggmlc.visualize(graph, output_path="model.html", format="html") # Interactive browser HTML
```

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

# 1. Compile Flax forward function directly to a GGUF file
model_path = ggmlc.compile(
    model=forward,
    sample_inputs=(jnp.ones((1, 32)),),
    output="flax_classifier.gguf",
    model_name="flax_classifier",
)

# 2. Run high-speed native inference
runner = ggmlc.load(model_path)
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
