"""JAX frontend for ggmlc."""

try:
    import jax

    if (
        hasattr(jax, "extend")
        and hasattr(jax.extend, "core")
        and not hasattr(jax.core, "get_opaque_trace_state")
    ):
        jax.core.get_opaque_trace_state = getattr(jax.extend.core, "get_opaque_trace_state", None)
except (AttributeError, ImportError):
    pass

from ggmlc.frontend.jax.exporter import export_jax_fn
from ggmlc.frontend.jax.importer import import_jaxpr

__all__ = ["export_jax_fn", "import_jaxpr"]
