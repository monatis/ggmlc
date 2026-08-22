import ggmlc
import jax
import jax.numpy as jnp
import numpy as np
from ggmlc.validation.numerical import check_numerical_accuracy, cosine_similarity

from examples.models.flax_models import (
    FlaxFullTransformer,
    FlaxMLPClassifier,
    FlaxTransformerLayer,
)


def test_flax_mlp_classifier_e2e():
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    model = FlaxMLPClassifier(hidden_dim=64, num_classes=10)
    x_sample = jax.random.normal(k1, (2, 32), dtype=jnp.float32)
    params = model.init(k2, x_sample)

    def forward(x):
        return model.apply(params, x)

    # Reference JAX forward pass
    ref_out = np.asarray(forward(x_sample))

    # 1. Compile Flax model with ggmlc
    x_np = np.asarray(x_sample)
    gguf_bytes = ggmlc.compile(
        model=forward,
        sample_inputs=(x_np,),
        model_name="flax_mlp",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    # 2. Execute with high-performance native runner
    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    # 3. Numerical verification against JAX golden truth
    res = check_numerical_accuracy(ref_out, out_np, atol=1e-4, rtol=1e-3)
    cos_sim = cosine_similarity(ref_out, out_np)
    assert res.passed, f"Flax MLP parity failed: {res.message}, cos_sim={cos_sim}"
    assert cos_sim > 0.9999


def test_flax_transformer_layer_e2e():
    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key)

    model = FlaxTransformerLayer(dim=32, num_heads=2, mlp_dim=64)
    x_sample = jax.random.normal(k1, (1, 8, 32), dtype=jnp.float32)
    params = model.init(k2, x_sample)

    def forward(x):
        return model.apply(params, x)

    # Reference JAX forward pass
    ref_out = np.asarray(forward(x_sample))

    # Compile with ggmlc
    x_np = np.asarray(x_sample)
    gguf_bytes = ggmlc.compile(
        model=forward,
        sample_inputs=(x_np,),
        model_name="flax_transformer_layer",
    )

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.999, f"Flax Transformer Layer cosine similarity too low: {cos_sim}"


def test_flax_full_multi_layer_transformer_e2e():
    key = jax.random.PRNGKey(999)
    k1, k2 = jax.random.split(key)

    model = FlaxFullTransformer(num_layers=2, dim=32, num_heads=2, mlp_dim=64)
    x_sample = jax.random.normal(k1, (1, 4, 32), dtype=jnp.float32)
    params = model.init(k2, x_sample)

    def forward(x):
        return model.apply(params, x)

    ref_out = np.asarray(forward(x_sample))

    x_np = np.asarray(x_sample)
    gguf_bytes = ggmlc.compile(
        model=forward,
        sample_inputs=(x_np,),
        model_name="flax_full_transformer",
    )

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.999, f"Flax Full Transformer cosine similarity too low: {cos_sim}"
