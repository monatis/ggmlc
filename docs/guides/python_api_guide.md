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
    pipeline: Any = None,
    tasks: str | list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
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
- **`pipeline`**: Optional multimodal preprocessor (e.g. `VisionPreprocessor` or `BPETokenizer`) whose specification metadata is serialized directly into GGUF headers for standalone runtime execution.
- **`tasks`**: Explicit declared task or list of tasks (`"classification"`, `["embedding", "similarity"]`, `"text-generation"`), embedded into `"ggmlc.tasks"` for task-aware output formatting and capability checks.
- **`extra_metadata`**: Optional dictionary of custom key-value metadata to embed into GGUF headers.
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

# Persistent state management (for KV-cache and recurrent states)
runner.set_state("state_tensor", state_numpy)
current_state = runner.get_state("state_tensor")
runner.reset_state()
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

---

## 3. Multimodal Preprocessors & Tokenizers

`ggmlc.pipeline` provides automated introspection builders for vision and text preprocessing that serialize directly into standard GGUF metadata.

### Vision Preprocessors (`VisionPreprocessor`)
Automatically introspects reference parameters (crop size, shortest-edge resize, mean, std, interpolation):
```python
from PIL import Image
import torchvision.models as models
from ggmlc.pipeline import VisionPreprocessor

# 1. Introspect from Torchvision Weights enum
pre_resnet = VisionPreprocessor.from_torchvision(models.ResNet50_Weights.DEFAULT)
pixel_tensor = pre_resnet(Image.open("sample.jpg"))  # (1, 3, 224, 224) np.float32

# 2. Introspect from Hugging Face Model ID
pre_clip = VisionPreprocessor.from_huggingface("openai/clip-vit-base-patch32")
clip_tensor = pre_clip(Image.open("sample.jpg"))

# 3. Embed directly into GGUF container during compilation
ggmlc.compile(
    resnet_model,
    (torch.randn(1, 3, 224, 224),),
    output="resnet50_vision.gguf",
    pipeline=pre_resnet,
    tasks=["classification"],
)
```

### Tokenizers (`BPETokenizer`)
Fast BPE tokenization with native C++ runtime support:
```python
from ggmlc.pipeline.tokenizer import BPETokenizer

# Load from Hugging Face model repository
tokenizer = BPETokenizer.from_huggingface("HuggingFaceTB/SmolLM2-135M-Instruct")

# 1. Raw prompt encoding & decoding
token_ids = tokenizer.encode("The capital of France is")
text = tokenizer.decode(token_ids)

# 2. Chat template formatting (ChatML, Gemma, Llama-3)
formatted = tokenizer.apply_chat_template("What is the capital of France?", system_msg="You are helpful.")
chat_ids = tokenizer.encode(formatted, add_special_tokens=False)
```
