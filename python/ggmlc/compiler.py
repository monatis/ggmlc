"""Unified high-level compilation and code generation API for ggmlc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ggmlc.codegen.cpp import generate_cpp_project
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.quantization import quantize_graph_parameters
from ggmlc.serialization.gguf import serialize_ggml_graph
from ggmlc.transforms import create_standard_optimization_pipeline


def compile(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any] | None = None,
    dynamic_shapes: tuple[dict[int, Any], ...] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    output: str | Path | None = None,
    **kwargs: Any,
) -> bytes:
    """Compiles a PyTorch or JAX neural network model into a standard GGUF v3 binary container.

    Args:
        model: PyTorch model (nn.Module or ExportedProgram) or JAX function/jaxpr.
        sample_inputs: Tuple of sample input tensors matching model input signature.
        dynamic_shapes: Optional dynamic shape constraints (e.g. torch.export.Dim).
        model_name: Name identifier for the compiled model graph.
        enable_optimizations: If True, applies standard IR graph optimizations (CF, DCE).
        enable_fusion: If True, lowers composite subgraphs into high-performance fused ops.
        fusion_options: Optional granular control over specific fusion patterns.
        quantize: Optional quantization format ('q4_0', 'q8_0', DType.Q4_0, DType.Q8_0).
        output: Optional file path to save the compiled .gguf binary container.

    Returns:
        Serialized GGUF v3 binary bytes.
    """
    # 1. Ingest model into Canonical IR Graph
    canonical_graph: Graph
    if isinstance(model, Graph):
        canonical_graph = model
    elif hasattr(model, "graph_module") or hasattr(model, "module"):  # ExportedProgram or nn.Module
        from ggmlc.frontend.pytorch import export_torch_model

        if sample_inputs is None:
            raise ValueError("sample_inputs must be provided when compiling a PyTorch model.")
        inputs_tuple = tuple(sample_inputs) if isinstance(sample_inputs, list) else sample_inputs
        exported = export_torch_model(
            model, inputs_tuple, dynamic_shapes=dynamic_shapes, model_name=model_name
        )
        canonical_graph = exported.main_graph
    elif callable(model) and not hasattr(model, "parameters"):  # JAX function or callable
        import jax

        from ggmlc.frontend.jax import import_jaxpr

        if sample_inputs is None:
            raise ValueError("sample_inputs must be provided when compiling a JAX function.")
        inputs_tuple = tuple(sample_inputs) if isinstance(sample_inputs, list) else sample_inputs
        jaxpr = jax.make_jaxpr(model)(*inputs_tuple)
        canonical_graph = import_jaxpr(jaxpr, graph_name=model_name, **kwargs)
    else:
        # Fallback PyTorch export attempt
        from ggmlc.frontend.pytorch import export_torch_model

        if sample_inputs is None:
            raise ValueError("sample_inputs must be provided for model compilation.")
        inputs_tuple = tuple(sample_inputs) if isinstance(sample_inputs, list) else sample_inputs
        exported = export_torch_model(
            model, inputs_tuple, dynamic_shapes=dynamic_shapes, model_name=model_name
        )
        canonical_graph = exported.main_graph

    canonical_graph.name = model_name

    # 2. Run Canonical IR Graph Optimizations
    if enable_optimizations:
        pipeline = create_standard_optimization_pipeline()
        opt_result = pipeline.run(canonical_graph)
        canonical_graph = opt_result.graph

    # 3. Lower to GGML Dialect
    ggml_graph = lower_to_ggml(
        canonical_graph,
        enable_fusion=enable_fusion,
        fusion_options=fusion_options,
    )

    # 4. Apply Block Quantization (Optional)
    if quantize is not None:
        target_dtype = DType.from_str(str(quantize)) if isinstance(quantize, str) else quantize
        ggml_graph, _ = quantize_graph_parameters(ggml_graph, target_dtype=target_dtype)

    # 5. Serialize to standard GGUF v3 container
    gguf_bytes = serialize_ggml_graph(ggml_graph)

    # 6. Save to disk if output path is requested
    if output is not None:
        out_path = Path(output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(gguf_bytes)

    return gguf_bytes


def codegen(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any],
    output_dir: str | Path,
    dynamic_shapes: tuple[dict[int, Any], ...] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: dict[str, bool] | None = None,
) -> Path:
    """Transpiles a neural network graph into a standalone, human-readable C++ project.

    Generates <ModelName>.h (weights & build_graph), ggmlc_main.cpp (runner),
    and CMakeLists.txt ready for native compilation.

    Args:
        model: PyTorch model or exported program.
        sample_inputs: Sample input tensors matching model input signature.
        output_dir: Destination directory for the generated C++ project.
        dynamic_shapes: Optional dynamic shape constraints.
        model_name: Base class/file name identifier.
        enable_optimizations: If True, applies standard IR optimizations.
        enable_fusion: If True, lowers composite subgraphs to fused ops.
        fusion_options: Optional granular fusion options.

    Returns:
        Path to the generated project directory.
    """
    from ggmlc.frontend.pytorch import export_torch_model

    inputs_tuple = tuple(sample_inputs) if isinstance(sample_inputs, list) else sample_inputs
    exported = export_torch_model(
        model, inputs_tuple, dynamic_shapes=dynamic_shapes, model_name=model_name
    )

    return generate_cpp_project(
        exported_program=exported,
        output_dir=output_dir,
        model_name=model_name,
        enable_fusion=enable_fusion,
        fusion_options=fusion_options,
    )
