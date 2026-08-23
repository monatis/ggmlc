# ggmlc Python High-Performance API & User Guide

`ggmlc` provides a modern, unified, framework-agnostic Python API for compiling, optimizing, quantizing, and executing neural network models from **PyTorch**, **JAX**, and **Flax** directly onto CPU and **NVIDIA CUDA GPU** backends using GGML.

---

## 1. Quickstart: Compiling & Running on CPU and GPU

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

# 3. Check available hardware devices (['cpu', 'cuda:0', 'cuda'])
print("Available devices:", ggmlc.get_available_devices())

# 4. Load into high-performance native C++ nanobind runtime on CPU or GPU
runner_cpu = ggmlc.load(model_path, device="cpu", n_threads=4)
runner_gpu = ggmlc.load(model_path, device="cuda")  # Executes natively on NVIDIA GPU

# 5. Evaluate with zero-copy numpy buffers
output = runner_gpu(example_input.numpy())
print("GPU Runner device:", runner_gpu.device)
print("Output shape:", output.shape)
```

---

## 2. Core Python APIs

### `ggmlc.get_available_devices`
Returns the list of hardware execution devices discovered and supported by the current runtime:
```python
devices = ggmlc.get_available_devices()
# Example output: ['cpu', 'cuda:0', 'cuda']
```

---

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
    device: str = "cpu",
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
- **`device`**: Hardware target when `return_runner=True` (`"cpu"`, `"cuda"`, `"cuda:0"`, `"auto"`).

---

### `ggmlc.compile_to_bytes`
Helper function for compiling and returning raw in-memory GGUF bytes when needed:
```python
gguf_bytes = ggmlc.compile_to_bytes(model, sample_inputs=(x,), model_name="my_model")
```

---

### `ggmlc.load`
Loads a compiled model from a `.gguf` file path or in-memory `bytes` into a `ModelRunner`.

```python
# CPU execution with multi-threading
runner_cpu = ggmlc.load("model.gguf", device="cpu", n_threads=4)

# GPU execution via NVIDIA CUDA
runner_gpu = ggmlc.load("model.gguf", device="cuda")

# Auto device selection (CUDA if available, falling back to CPU)
runner_auto = ggmlc.load("model.gguf", device="auto")
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
Generates an independent, standalone C++ model library and executable with dual CPU and CUDA backend support:

```python
ggmlc.codegen(
    model=model,
    sample_inputs=(example_input,),
    output_dir="./generated_cpp",
    model_name="MyModel",
)
```

Generates:
- `MyModel.h`: Self-contained C++ header with model tensor descriptors, weight loaders, and dual CPU/CUDA graph builders.
- `ggmlc_main.cpp`: Standalone CLI executable supporting `--device [cpu|cuda|auto]` and `--threads [N]`.
- `CMakeLists.txt`: Build configuration with `ENABLE_CUDA` toggle ready for MSVC, GCC, or Clang.

---

### `ggmlc.visualize`
Generates diagrammatic visualizations of Canonical IR or Lowered GGML execution graphs:

```python
# Export pure-Python rendered PNG, SVG, or interactive HTML with zoom/pan
ggmlc.visualize(graph, output_path="model_graph.png")  # Render to PNG image via mermaidx
ggmlc.visualize(graph, output_path="model_graph.svg")  # Render to vector SVG
ggmlc.visualize(graph, output_path="model_graph.html") # Interactive browser visualization
```
