import jax
import jax.numpy as jnp
import numpy as np
from ggmlc.frontend.jax import export_jax_fn
from ggmlc.ir.tensor import StorageClass


def test_jax_frontend_export():
    def simple_fn(a, b):
        return jnp.sin(a) + jnp.cos(b)

    x = np.ones((4, 8), dtype=np.float32)
    y = np.ones((4, 8), dtype=np.float32)

    model = export_jax_fn(simple_fn, (x, y), input_names=["a", "b"], model_name="trig_model")
    g = model.main_graph

    assert len(g.inputs) == 2
    assert len(g.outputs) == 1
    assert g.get_tensor(g.inputs[0]).name == "a"
    assert g.get_tensor(g.inputs[1]).name == "b"
    assert g.get_tensor(g.outputs[0]).storage == StorageClass.OUTPUT
    g.validate_invariants()


def test_jax_frontend_params_and_inlining():
    def mlp(x, w, b):
        return jax.nn.relu(jnp.dot(x, w) + b)

    x = np.random.randn(2, 16).astype(np.float32)
    w = np.random.randn(16, 32).astype(np.float32)
    b = np.random.randn(32).astype(np.float32)

    model = export_jax_fn(
        mlp,
        (x, w, b),
        input_names=["x", "w", "b"],
        params={"w": w, "b": b},
        model_name="mlp_model",
    )
    g = model.main_graph

    assert len(g.inputs) == 1
    params = [tid for tid in g.parameters if g.get_tensor(tid).storage == StorageClass.PARAMETER]
    assert len(params) == 2
    assert g.get_tensor(g.inputs[0]).name == "x"
    g.validate_invariants()
