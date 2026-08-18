import jax
import jax.numpy as jnp
import numpy as np
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.jax import export_jax_fn
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl


def test_jax_elementwise_e2e():
    def compute(x, y):
        return (x + y) * 2.5

    x = np.random.randn(4, 16).astype(np.float32)
    y = np.random.randn(4, 16).astype(np.float32)

    # 1. Reference output from JAX
    ref = np.array(compute(jnp.array(x), jnp.array(y)))

    # 2. Export & Lower & Serialize
    model = export_jax_fn(compute, (x, y), input_names=["x", "y"], model_name="jax_elem")
    ggml_graph = lower_to_ggml(model.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 3. Execute with generic C++ GGML runtime
    out_id = model.main_graph.outputs[0]
    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={"x": x, "y": y},
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    res = check_numerical_accuracy(ref, ggml_out, atol=1e-4)
    assert res.passed, f"JAX differential test failed: {res.message}"


def test_jax_mlp_layer_e2e():
    def mlp_layer(x, w, b):
        return jax.nn.relu(jnp.dot(x, w) + b)

    x = np.random.randn(2, 32).astype(np.float32)
    w = np.random.randn(32, 64).astype(np.float32)
    b = np.random.randn(64).astype(np.float32)

    # 1. Reference output from JAX
    ref = np.array(mlp_layer(jnp.array(x), jnp.array(w), jnp.array(b)))

    # 2. Export & Lower & Serialize
    model = export_jax_fn(
        mlp_layer,
        (x, w, b),
        input_names=["x", "w", "b"],
        params={"w": w, "b": b},
        model_name="jax_mlp",
    )
    ggml_graph = lower_to_ggml(model.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 3. Execute with generic C++ GGML runtime
    out_id = model.main_graph.outputs[0]
    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={"x": x},
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    res = check_numerical_accuracy(ref, ggml_out, atol=1e-4)
    assert res.passed, f"JAX MLP differential test failed: {res.message}"
