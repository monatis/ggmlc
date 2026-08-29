"""Cross-backend differential integration tests for Keras 3 (PyTorch vs JAX backends).

Verifies that identical neural network architectures and weights compiled from both
PyTorch (`KERAS_BACKEND=torch` -> `torch.export`) and JAX (`KERAS_BACKEND=jax` -> `jax.make_jaxpr`)
produce numerically equivalent GGML execution graphs and runtime predictions.
"""

import json
import os
import subprocess
import sys

import numpy as np
from ggmlc.validation.numerical import check_numerical_accuracy, cosine_similarity

PYTHON_EXE = sys.executable

WORKER_CODE = """
import os
import sys
import json
import numpy as np

backend = sys.argv[1]
model_type = sys.argv[2]
weights_path = sys.argv[3]

os.environ["KERAS_BACKEND"] = backend
import keras
import ggmlc
from ggmlc.runtime.runner import load

with open(weights_path, "r") as f:
    w_data = json.load(f)

sample_in = np.array(w_data["sample_in"], dtype=np.float32)

if model_type == "mlp":
    inp = keras.Input(shape=(32,), name="input_x")
    d1 = keras.layers.Dense(64, activation="relu", name="dense1")
    d2 = keras.layers.Dense(16, activation="softmax", name="dense2")
    x = d2(d1(inp))
    model = keras.Model(inputs=inp, outputs=x, name="mlp")
    d1.set_weights([np.array(w_data["w1"], dtype=np.float32), np.array(w_data["b1"], dtype=np.float32)])
    d2.set_weights([np.array(w_data["w2"], dtype=np.float32), np.array(w_data["b2"], dtype=np.float32)])

elif model_type == "conv_block":
    inp = keras.Input(shape=(28, 28, 3), name="input_img")
    c1 = keras.layers.Conv2D(16, (3, 3), padding="same", name="conv1")
    r1 = keras.layers.ReLU(name="relu1")
    p1 = keras.layers.MaxPooling2D((2, 2), name="pool1")
    fl = keras.layers.Flatten(name="flat")
    fc = keras.layers.Dense(10, name="fc")
    x = fc(fl(p1(r1(c1(inp)))))
    model = keras.Model(inputs=inp, outputs=x, name="conv_block")
    c1.set_weights([np.array(w_data["conv_w"], dtype=np.float32), np.array(w_data["conv_b"], dtype=np.float32)])
    fc.set_weights([np.array(w_data["fc_w"], dtype=np.float32), np.array(w_data["fc_b"], dtype=np.float32)])

elif model_type == "resnet_block":
    inp = keras.Input(shape=(16, 16, 8), name="res_in")
    c1 = keras.layers.Conv2D(8, (3, 3), padding="same", name="res_c1")
    r1 = keras.layers.ReLU(name="res_r1")
    c2 = keras.layers.Conv2D(8, (3, 3), padding="same", name="res_c2")
    x = keras.layers.Add(name="res_add")([inp, c2(r1(c1(inp)))])
    model = keras.Model(inputs=inp, outputs=x, name="resnet_block")
    c1.set_weights([np.array(w_data["c1_w"], dtype=np.float32), np.array(w_data["c1_b"], dtype=np.float32)])
    c2.set_weights([np.array(w_data["c2_w"], dtype=np.float32), np.array(w_data["c2_b"], dtype=np.float32)])

elif model_type == "layernorm_block":
    inp = keras.Input(shape=(16, 32), name="seq_in")
    d1 = keras.layers.Dense(32, name="dense")
    ln = keras.layers.LayerNormalization(name="ln")
    x = ln(d1(inp))
    model = keras.Model(inputs=inp, outputs=x, name="layernorm_block")
    d1.set_weights([np.array(w_data["d_w"], dtype=np.float32), np.array(w_data["d_b"], dtype=np.float32)])
    ln.set_weights([np.array(w_data["ln_gamma"], dtype=np.float32), np.array(w_data["ln_beta"], dtype=np.float32)])

else:
    raise ValueError(f"Unknown model_type: {model_type}")

# Compute native reference
ref_out = model(sample_in, training=False)
if hasattr(ref_out, "detach"):
    ref_out_np = ref_out.detach().cpu().numpy()
elif hasattr(ref_out, "numpy"):
    ref_out_np = ref_out.numpy()
else:
    ref_out_np = np.asarray(ref_out)

# Compile to GGML
if backend == "torch":
    import torch
    torch_in = torch.from_numpy(sample_in)
    gguf_bytes = ggmlc.compile_to_bytes(model, (torch_in,), enable_fusion=True)
else:
    import jax
    import jax.numpy as jnp
    jax_in = jnp.array(sample_in)
    gguf_bytes = ggmlc.compile_to_bytes(lambda x: model(x, training=False), (jax_in,), enable_fusion=True)

runner = load(gguf_bytes, device="cpu")
ggmlc_out = runner(sample_in)
if not isinstance(ggmlc_out, np.ndarray):
    ggmlc_out = ggmlc_out.numpy()

res = {
    "backend": backend,
    "ref_out": ref_out_np.tolist(),
    "ggmlc_out": ggmlc_out.tolist(),
    "gguf_size": len(gguf_bytes),
}
print("===JSON_OUT_START===")
print(json.dumps(res))
print("===JSON_OUT_END===")
"""


def _run_backend_subprocess(backend: str, model_type: str, weights_path: str) -> dict:
    cmd = [PYTHON_EXE, "-c", WORKER_CODE, backend, model_type, weights_path]
    env = os.environ.copy()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    assert res.returncode == 0, (
        f"Error in {backend} {model_type}:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    )
    stdout = res.stdout
    assert "===JSON_OUT_START===" in stdout and "===JSON_OUT_END===" in stdout, (
        f"Malformed output: {stdout}"
    )
    json_str = stdout.split("===JSON_OUT_START===")[1].split("===JSON_OUT_END===")[0].strip()
    return json.loads(json_str)


def test_cross_backend_mlp_differential(tmp_path):
    """Verifies MLP architecture compiled from PyTorch vs JAX backends."""
    np.random.seed(42)
    w_data = {
        "sample_in": np.random.randn(1, 32).astype(np.float32).tolist(),
        "w1": np.random.randn(32, 64).astype(np.float32).tolist(),
        "b1": np.random.randn(64).astype(np.float32).tolist(),
        "w2": np.random.randn(64, 16).astype(np.float32).tolist(),
        "b2": np.random.randn(16).astype(np.float32).tolist(),
    }
    weights_path = str(tmp_path / "mlp_weights.json")
    with open(weights_path, "w") as f:
        json.dump(w_data, f)

    res_torch = _run_backend_subprocess("torch", "mlp", weights_path)
    res_jax = _run_backend_subprocess("jax", "mlp", weights_path)

    torch_ggml_out = np.array(res_torch["ggmlc_out"])
    jax_ggml_out = np.array(res_jax["ggmlc_out"])
    torch_ref = np.array(res_torch["ref_out"])
    jax_ref = np.array(res_jax["ref_out"])

    # Check cross-backend GGML parity
    cos_sim = cosine_similarity(torch_ggml_out, jax_ggml_out)
    acc = check_numerical_accuracy(torch_ggml_out, jax_ggml_out, atol=1e-4, rtol=1e-4)
    assert cos_sim > 0.9999, f"Cross-backend cosine similarity low: {cos_sim}"
    assert acc.passed, f"Cross-backend numerical accuracy failed: max_diff={acc.max_diff}"

    # Check parity with reference
    assert cosine_similarity(torch_ggml_out, torch_ref) > 0.9999
    assert cosine_similarity(jax_ggml_out, jax_ref) > 0.9999


def test_cross_backend_conv_block_differential(tmp_path):
    """Verifies Conv2D + ReLU + MaxPool2D + Dense compiled from PyTorch vs JAX."""
    np.random.seed(1337)
    w_data = {
        "sample_in": np.random.randn(1, 28, 28, 3).astype(np.float32).tolist(),
        "conv_w": np.random.randn(3, 3, 3, 16).astype(np.float32).tolist(),
        "conv_b": np.random.randn(16).astype(np.float32).tolist(),
        "fc_w": np.random.randn(3136, 10).astype(np.float32).tolist(),
        "fc_b": np.random.randn(10).astype(np.float32).tolist(),
    }
    weights_path = str(tmp_path / "conv_weights.json")
    with open(weights_path, "w") as f:
        json.dump(w_data, f)

    res_torch = _run_backend_subprocess("torch", "conv_block", weights_path)
    res_jax = _run_backend_subprocess("jax", "conv_block", weights_path)

    torch_ggml_out = np.array(res_torch["ggmlc_out"])
    jax_ggml_out = np.array(res_jax["ggmlc_out"])

    cos_sim = cosine_similarity(torch_ggml_out, jax_ggml_out)
    acc = check_numerical_accuracy(torch_ggml_out, jax_ggml_out, atol=1e-3, rtol=1e-3)
    assert cos_sim > 0.9999, f"Conv block cross-backend cosine similarity low: {cos_sim}"
    assert acc.passed, f"Conv block numerical accuracy failed: max_diff={acc.max_diff}"


def test_cross_backend_resnet_residual_differential(tmp_path):
    """Verifies Residual Skip connection Block compiled from PyTorch vs JAX."""
    np.random.seed(7)
    w_data = {
        "sample_in": np.random.randn(1, 16, 16, 8).astype(np.float32).tolist(),
        "c1_w": np.random.randn(3, 3, 8, 8).astype(np.float32).tolist(),
        "c1_b": np.random.randn(8).astype(np.float32).tolist(),
        "c2_w": np.random.randn(3, 3, 8, 8).astype(np.float32).tolist(),
        "c2_b": np.random.randn(8).astype(np.float32).tolist(),
    }
    weights_path = str(tmp_path / "resnet_weights.json")
    with open(weights_path, "w") as f:
        json.dump(w_data, f)

    res_torch = _run_backend_subprocess("torch", "resnet_block", weights_path)
    res_jax = _run_backend_subprocess("jax", "resnet_block", weights_path)

    torch_ggml_out = np.array(res_torch["ggmlc_out"])
    jax_ggml_out = np.array(res_jax["ggmlc_out"])

    cos_sim = cosine_similarity(torch_ggml_out, jax_ggml_out)
    acc = check_numerical_accuracy(torch_ggml_out, jax_ggml_out, atol=1e-3, rtol=1e-3)
    assert cos_sim > 0.9999, f"ResNet block cross-backend cosine similarity low: {cos_sim}"
    assert acc.passed, f"ResNet block numerical accuracy failed: max_diff={acc.max_diff}"


def test_cross_backend_layernorm_differential(tmp_path):
    """Verifies Dense + LayerNormalization block compiled from PyTorch vs JAX."""
    np.random.seed(99)
    w_data = {
        "sample_in": np.random.randn(1, 16, 32).astype(np.float32).tolist(),
        "d_w": np.random.randn(32, 32).astype(np.float32).tolist(),
        "d_b": np.random.randn(32).astype(np.float32).tolist(),
        "ln_gamma": np.random.randn(32).astype(np.float32).tolist(),
        "ln_beta": np.random.randn(32).astype(np.float32).tolist(),
    }
    weights_path = str(tmp_path / "ln_weights.json")
    with open(weights_path, "w") as f:
        json.dump(w_data, f)

    res_torch = _run_backend_subprocess("torch", "layernorm_block", weights_path)
    res_jax = _run_backend_subprocess("jax", "layernorm_block", weights_path)

    torch_ggml_out = np.array(res_torch["ggmlc_out"])
    jax_ggml_out = np.array(res_jax["ggmlc_out"])

    cos_sim = cosine_similarity(torch_ggml_out, jax_ggml_out)
    acc = check_numerical_accuracy(torch_ggml_out, jax_ggml_out, atol=1e-3, rtol=1e-3)
    assert cos_sim > 0.9999, f"LayerNorm cross-backend cosine similarity low: {cos_sim}"
    assert acc.passed, f"LayerNorm numerical accuracy failed: max_diff={acc.max_diff}"
