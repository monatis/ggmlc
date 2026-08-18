from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

from ggmlc.frontend.jax.importer import import_jaxpr
from ggmlc.ir.model import Model


def export_jax_fn(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    input_names: Sequence[str] | None = None,
    params: dict[str, np.ndarray] | None = None,
    model_name: str = "jax_model",
) -> Model:
    """Exports a JAX function into a ggmlc Model containing Canonical IR graphs."""
    closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
    g = import_jaxpr(
        closed_jaxpr,
        graph_name="main",
        input_names=input_names,
        params=params,
    )
    m = Model(name=model_name)
    m.add_graph(g, is_main=True)
    return m
