from __future__ import annotations

from typing import Any

import torch
from torch.export import export

from ggmlc.frontend.pytorch.importer import import_exported_program
from ggmlc.ir.model import Model
from ggmlc.transforms import create_standard_optimization_pipeline


def export_torch_model(
    model: torch.nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    model_name: str = "model",
    optimize: bool = True,
) -> Model:
    """Exports a PyTorch model into a ggmlc Model containing Canonical IR graphs."""
    model.eval()
    ep = export(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    g = import_exported_program(ep, graph_name="main")
    if optimize:
        pipeline = create_standard_optimization_pipeline()
        g = pipeline(g)
    m = Model(name=model_name)
    m.add_graph(g, is_main=True)
    return m
