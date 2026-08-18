from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import numpy as np

from ggmlc.frontend.jax.importer import import_jaxpr
from ggmlc.ir.graph import Graph
from ggmlc.ir.model import Model


def export_jax_fn(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    input_names: Optional[Sequence[str]] = None,
    params: Optional[Dict[str, np.ndarray]] = None,
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
