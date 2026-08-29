# ggmlc Developer & Contributor Guide

Welcome to the `ggmlc` developer guide. This document provides step-by-step instructions for extending the compiler frontend, Canonical IR, dialect lowering, optimization passes, quantization algorithms, C++ generic runtime, and testing infrastructure.

---

## 1. Core Engineering Principles

1. **A neural network is a semantic tensor program**: Graphs are purely functional DAGs where nodes are operators and edges are value tensors.
2. **Framework Independence**: Frontends (`torch.export`, JAX `jaxpr`) produce Canonical IR. The dialect lowerings translate Canonical IR to target semantics without frontend leakage.
3. **Small Vertical Slices**: Every feature/op must have:
   - IR definition in `python/ggmlc/ir/op.py`
   - Shape inference rule in `python/ggmlc/ir/shape.py` or op schema
   - Frontend importer rule in `python/ggmlc/frontend/`
   - Target dialect lowering in `python/ggmlc/dialect/ggml/lowering.py`
   - OpCode / kernel handling in C++ runtime `runtime/src/executor.cpp`
   - Python unit test in `tests/ops/`
   - Differential numerical test in `tests/numerical/` or `tests/e2e/`
4. **Correctness Before Optimization**: Differential testing against reference PyTorch and JAX runs is the golden source of truth.

---

## 2. Step-by-Step: Adding a New Operator

### Step 1: Define the Canonical OpCode & Schema
In `python/ggmlc/ir/op.py`:
```python
class OpCode(Enum):
    # ...
    MY_NEW_OP = "my_new_op"

@dataclass
class OpSchema:
    opcode: OpCode
    min_inputs: int
    max_inputs: int
    num_outputs: int
    infer_shape_fn: Callable[..., list[Shape]]
```

### Step 2: Implement Shape Inference
In `python/ggmlc/ir/shape.py`:
```python
def infer_my_new_op_shape(inputs: list[Tensor], attributes: dict[str, Any]) -> list[Shape]:
    # Compute output shape from input static and symbolic dimensions
    in_shape = inputs[0].shape
    return [in_shape]
```

### Step 3: Add Frontend Importer Mapping
For PyTorch (`python/ggmlc/frontend/pytorch/importer.py`):
```python
# In target dispatch map
if target_name in ("aten.my_new_op.default", "aten.my_new_op.Tensor"):
    in_tensor = self._get_tensor(node.args[0])
    out_tensor = graph.add_tensor(shape=infer_shape, dtype=in_tensor.dtype)
    graph.add_node(OpCode.MY_NEW_OP, inputs=[in_tensor.id], outputs=[out_tensor.id])
```

### Step 4: Add GGML Dialect Lowering
In `python/ggmlc/dialect/ggml/lowering.py`:
```python
def _lower_my_new_op(node: Operation, in_graph: Graph, out_graph: Graph, tensor_map: dict):
    in_id = tensor_map[node.inputs[0]]
    out_id = tensor_map[node.outputs[0]]
    out_graph.add_node(
        opcode=OpCode.MY_NEW_OP,
        inputs=[in_id],
        outputs=[out_id],
        attributes={"attr_key": node.attributes.get("attr_key", 0)},
    )
```

### Step 5: Add C++ Runtime Kernel Handling
In `runtime/src/executor.cpp`:
```cpp
case GGML_OP_MY_NEW_OP: {
    struct ggml_tensor* in0 = ggml_tensors_[op.inputs[0]];
    if (is_cuda_) {
        // Native GPU kernel / operation emission
        result = ggml_my_new_op(ctx_, in0);
    } else {
        // CPU execution (multi-threaded fallback or custom kernel)
        result = ggml_my_new_op(ctx_, in0);
    }
    break;
}
```

### Step 6: Write Python Unit & Numerical Tests
In `tests/ops/test_my_new_op.py` and `tests/numerical/`:
```python
def test_my_new_op_numerical_parity():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.my_new_op(x)

    model = M().eval()
    x = torch.randn(2, 16, dtype=torch.float32)
    ref_out = model(x).detach().numpy()

    # Compile and evaluate across CPU and CUDA
    runner_cpu = ggmlc.compile(model, (x,), return_runner=True, device="cpu")
    cpu_out = runner_cpu(x.numpy())
    assert np.allclose(ref_out, cpu_out, atol=1e-4)

    if any(d.startswith("cuda") for d in ggmlc.get_available_devices()):
        runner_gpu = ggmlc.compile(model, (x,), return_runner=True, device="cuda")
        gpu_out = runner_gpu(x.numpy())
        assert np.allclose(ref_out, gpu_out, atol=1e-4)
```

---

## 3. Step-by-Step: Adding an Optimization Pass

1. **Subclass `Pass`** in `python/ggmlc/transforms/`:
   ```python
   from ggmlc.transforms.base import Pass, GraphTransformResult, PassStats
   from ggmlc.ir.graph import Graph

   class MyOptimizationPass(Pass):
       @property
       def name(self) -> str:
           return "my_optimization_pass"

       def run(self, graph: Graph) -> GraphTransformResult:
           # Build new optimized graph
           new_graph = Graph(graph.name)
           # ... apply pattern replacements ...
           return GraphTransformResult(graph=new_graph, stats=PassStats(...))
   ```
2. **Register in Pipeline** in `python/ggmlc/transforms/__init__.py`:
   Add the pass to `create_standard_optimization_pipeline()`.
3. **Add Tests** in `tests/transforms/test_transforms.py`.

---

## 4. Step-by-Step: Adding a New Quantization Format

1. **Add DType Enum**: In `python/ggmlc/ir/dtype.py`, declare the format enum and set `is_quantized = True`.
2. **Add GGML Type Mapping**: In `python/ggmlc/dialect/ggml/ops.py` and `lowering.py`, map the enum to the matching GGML type.
3. **Implement Block Quantizer**: In `python/ggmlc/quantization/quantize.py`, implement block encoding and decoding routines with bit packing.
4. **Register in Model Quantizer**: Update `quantize_graph_parameters()` in `python/ggmlc/quantization/model_quantizer.py`.
5. **Add Verification Tests**: In `tests/quantization/test_quantization.py`, add block accuracy tests (cosine similarity $> 0.98$) and end-to-end runtime execution tests.

---

## 5. Development & Testing Commands

### Python Testing
```powershell
# Run entire test suite (Windows)
pytest -v --ignore=tests/numerical/test_ggml_ops_differential.py

# Run E2E full model tests (ResNet, MiniLM, GPT-2, Qwen, BGE-M3)
pytest tests/e2e/test_full_models.py -v

# Run Native CUDA E2E tests on GPU
pytest tests/e2e/test_cuda_models.py -v

# Run JAX / Flax E2E tests
pytest tests/e2e/test_jax_flax_models.py -v

# Run Keras 3 JAX Production Model E2E tests (ResNet-50, MobileNetV3, ConvNeXt, DenseNet, EfficientNet)
pytest tests/e2e/test_keras_models.py -v

# Run KerasHub & Flax ViT E2E tests (BERT, DistilBERT, GPT-2, Flax ViT-B/16)
pytest tests/e2e/test_kerashub_models.py -v

# Run Vision Transformer (ViT) tests
pytest tests/e2e/test_vit_models.py -v

# Run Audio Seq2Seq (Whisper) tests
pytest tests/e2e/test_audio_models.py -v
```

### Multi-Framework Ingestion with Keras 3
Keras 3 provides a unified API whose models can be executed and traced across multiple backends:
- **JAX Backend (`KERAS_BACKEND=jax`)**: Traced directly via `jax.make_jaxpr` to standard `ClosedJaxpr` expressions and ingested by `ggmlc.frontend.jax`.
- **PyTorch Backend (`KERAS_BACKEND=torch`)**: Pure PyTorch modules exported via `torch.export` to Canonical IR.
- **Cross-Framework Verification**: Compile identical architectures from both backends to verify bit-level IR canonicalization and numerical parity.

### Continuous Benchmarking
```powershell
# Run benchmark suite on CPU
python examples/benchmarks/benchmark_suite.py --backend cpu --runs 3 --warmup 1 --output-md benchmark_cpu_report.md

# Run benchmark suite on NVIDIA CUDA GPU
python examples/benchmarks/benchmark_suite.py --backend cuda --runs 3 --warmup 1 --output-md benchmark_cuda_report.md
```

### Native Runtime Compilation

#### Windows Native CUDA Build (MSVC 2022 + CUDA 11.3 + Ninja)
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

cmake -B build-win-cuda -G Ninja -DGGMLC_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" -DPython_EXECUTABLE="C:\Users\ailabs\ggmlc\.venv\Scripts\python.exe"
cmake --build build-win-cuda -j8
```

#### Windows Native CPU Build (MSVC Visual Studio)
```powershell
cmake -B build-win -G "Visual Studio 17 2022" -A x64 -DGGMLC_ENABLE_CUDA=OFF -DPython_EXECUTABLE="C:\Users\ailabs\ggmlc\.venv\Scripts\python.exe"
cmake --build build-win --config Release -j8
```

#### Linux / WSL 2 Build (CPU & CUDA GPU)
```bash
wsl cmake -B build -DGGMLC_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
wsl cmake --build build -j8
```

### Code Quality & Linting
```powershell
ruff format python/ tests/
ruff check --fix python/ tests/
```
