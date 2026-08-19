# ggmlc Developer & Contributor Guide

Welcome to the `ggmlc` developer guide. This document provides step-by-step instructions for extending the compiler frontend, Canonical IR, dialect lowering, C++ generic runtime, and testing infrastructure.

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

Let's walk through an example of adding a new operator (e.g. `GELU` or `CONV2D`).

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
    actual_out = res[exp.main_graph.outputs[0]].reshape(ref_out.shape)

    cmp = check_numerical_accuracy(ref_out, actual_out, atol=1e-5)
    assert cmp.passed, cmp.message
```

---

## 3. Adding a New Architecture to Hub Models

When adding a full neural network model from Hugging Face / TorchVision:
1. Add the loader function in `examples/models/hub_models.py`:
   - Download or instantiate the pretrained model in `.eval()` mode.
   - Construct dummy inputs with appropriate shape and dtype.
   - If the raw model's forward returns custom dataclasses (like `BaseModelOutputWithPooling`), wrap it with a lightweight `nn.Module` to expose exact tensor outputs.
2. Add end-to-end test in `tests/e2e/test_full_models.py`:
   - Call `_verify_full_model_e2e(model, inputs, input_names, "model_name", atol=...)`.
3. Verify autoregressive generation parity if it is a Causal LM using `verify_generation_parity_with_pytorch`.

---

## 4. Development Commands & Workflow

```bash
# Run the entire test suite
pytest -v

# Run only e2e hub model tests
pytest tests/e2e/test_full_models.py -v

# Format Python code
ruff format python/ examples/ tests/

# Lint Python code
ruff check python/ examples/ tests/ --fix

# Rebuild C++ Runtime (WSL / Linux)
cmake -B build && cmake --build build -j$(nproc)
```
