"""Unified high-level compilation and code generation API for ggmlc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ggmlc.codegen.cpp import generate_cpp_project
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.quantization import quantize_graph_parameters
from ggmlc.transforms import create_standard_optimization_pipeline


def compile(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any] | None = None,
    output: str | Path | None = None,
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    return_runner: bool = False,
    **kwargs: Any,
) -> Path | bytes | Any:
    """Compiles a PyTorch or JAX neural network model into a standard GGUF v3 binary container.

    This function ingests the source model into Canonical IR, runs standard optimization
    passes (constant folding, dead-code elimination, redundant cast pruning), performs
    target dialect lowering to GGML, applies optional parameter quantization (Q4_0, Q8_0),
    and serializes the result into a standardized GGUF v3 file.

    Args:
        model: PyTorch model (nn.Module or ExportedProgram), JAX callable, or IR Graph.
        sample_inputs: Sample input tensors matching the model's forward signature.
        output: Optional file path to stream the .gguf binary container directly to disk.
        dynamic_shapes: Dynamic shape specifications (e.g. torch.export.Dim).
        model_name: Identifier name embedded in graph and GGUF metadata.
        enable_optimizations: If True, applies standard IR graph optimization passes.
        enable_fusion: If True, lowers composite subgraphs into high-performance fused ops.
        fusion_options: Optional granular flags for specific fusion patterns.
        quantize: Optional quantization format ('q4_0', 'q8_0', DType.Q4_0, DType.Q8_0).
        return_runner: If True, automatically loads and returns an instantiated ModelRunner.
        **kwargs: Additional framework-specific keyword arguments.

    Returns:
        Path to the saved .gguf file (if output is provided), ModelRunner (if return_runner=True),
        or raw GGUF v3 bytes (if output is None and return_runner=False).

    Example:
        >>> import ggmlc, torch, torchvision.models as models
        >>> model = models.resnet18().eval()
        >>> x = torch.randn(1, 3, 224, 224)
        >>> # Stream directly to disk
        >>> model_path = ggmlc.compile(model, (x,), output="resnet18.gguf")
        >>> runner = ggmlc.load(model_path)
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
        import_kwargs = {k: v for k, v in kwargs.items() if k not in ("device", "n_threads")}
        canonical_graph = import_jaxpr(jaxpr, graph_name=model_name, **import_kwargs)
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
        pipeline = create_standard_optimization_pipeline(
            enable_fusion=enable_fusion, options=fusion_options
        )
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

    # 5. Stream directly to file or serialize to memory
    from ggmlc.runtime.runner import load
    from ggmlc.serialization.gguf import save_to_gguf, serialize_to_gguf

    device = kwargs.get("device", "cpu")
    n_threads = kwargs.get("n_threads", 1)

    if output is not None:
        out_path = save_to_gguf(ggml_graph, output)
        if return_runner:
            return load(out_path, n_threads=n_threads, device=device)
        return out_path

    if return_runner:
        gguf_bytes = serialize_to_gguf(ggml_graph)
        return load(gguf_bytes, n_threads=n_threads, device=device)

    return serialize_to_gguf(ggml_graph)


def compile_to_bytes(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any] | None = None,
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    **kwargs: Any,
) -> bytes:
    """Compiles a model and returns in-memory GGUF v3 bytes."""
    res = compile(
        model=model,
        sample_inputs=sample_inputs,
        output=None,
        dynamic_shapes=dynamic_shapes,
        model_name=model_name,
        enable_optimizations=enable_optimizations,
        enable_fusion=enable_fusion,
        fusion_options=fusion_options,
        quantize=quantize,
        return_runner=False,
        **kwargs,
    )
    assert isinstance(res, bytes)
    return res


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
