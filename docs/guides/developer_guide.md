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
    struct ggml_tensor* in0 = tensors[node.inputs[0]];
    struct ggml_tensor* out = ggml_my_new_op(ctx, in0);
    tensors[node.outputs[0]] = out;
    break;
}
```

### Step 6: Write Python Unit & Numerical Tests
In `tests/ops/test_my_new_op.py`:
```python
def test_my_new_op_numerical_parity():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.my_new_op(x)

    model = M().eval()
    x = torch.randn(2, 16, dtype=torch.float32)
    ref_out = model(x).detach().numpy()

    exp = export_torch_model(model, (x,), model_name="my_new_op_test")
    g = lower_to_ggml(exp.main_graph)
    ser = serialize_ggml_graph(g)
    res = run_compiled_model_wsl(ser, {"x": x.numpy()}, [exp.main_graph.outputs[0]])
    out = res[exp.main_graph.outputs[0]].reshape(ref_out.shape)

    assert np.allclose(ref_out, out, atol=1e-4)
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

```bash
# Run entire test suite
pytest -v

# Run only quantization tests
pytest tests/quantization/ -v

# Run only optimization pass tests
pytest tests/transforms/ -v

# Format and lint code
ruff check python/ tests/
ruff format python/ tests/

# Rebuild C++ runtime (via WSL/Linux)
cmake --build build -j$(nproc)
```
