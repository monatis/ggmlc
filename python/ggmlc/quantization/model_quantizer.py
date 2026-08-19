"""Graph-level neural network parameter quantization."""

from __future__ import annotations

import numpy as np

from ggmlc.ir.graph import Graph
from ggmlc.ir.tensor import DType, StorageClass, Tensor
from ggmlc.quantization.quantize import quantize_q4_0, quantize_q8_0


def quantize_graph_parameters(
    graph: Graph,
    target_dtype: DType = DType.Q4_0,
    min_elements_to_quantize: int = 128,
) -> tuple[Graph, dict[str, int]]:
    """Quantizes 2D/3D parameter weights in the graph to the target quantization format.

    Returns:
      (quantized_graph, stats_dict)
    """
    if target_dtype not in (DType.Q4_0, DType.Q8_0):
        msg = f"Unsupported quantization dtype: {target_dtype}"
        raise ValueError(msg)

    from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph, GGMLTensorDef
    from ggmlc.dialect.ggml.ops import GGMLType

    if isinstance(graph, GGMLExecutionGraph):
        new_graph = GGMLExecutionGraph(name=f"{graph.name}_{target_dtype.name.lower()}")
        new_graph.symbol_table = list(graph.symbol_table)
    else:
        new_graph = Graph(name=f"{graph.name}_{target_dtype.name.lower()}")
        new_graph.parameters = list(getattr(graph, "parameters", []))
        new_graph.states = list(getattr(graph, "states", []))

    new_graph.inputs = list(graph.inputs)
    new_graph.outputs = list(graph.outputs)
    if hasattr(graph, "parameters"):
        new_graph.parameters = list(graph.parameters)
    new_graph.nodes = list(graph.nodes)

    orig_bytes = 0
    quant_bytes = 0
    tensors_quantized = 0

    for tid, tensor in graph.tensors.items():
        is_param = tensor.storage == StorageClass.PARAMETER
        has_f32 = (getattr(tensor, "dtype", None) == DType.F32) or (
            getattr(tensor, "ggml_type", None) == GGMLType.GGML_TYPE_F32
        )
        is_multi_d = (
            len(tensor.shape.dims) >= 2
            if hasattr(tensor, "shape")
            else (
                tensor.ne[1].evaluate({}) > 1
                or tensor.ne[2].evaluate({}) > 1
                or tensor.ne[3].evaluate({}) > 1
            )
        )

        if is_param and tensor.data is not None and has_f32 and is_multi_d:
            arr = np.array(tensor.data, dtype=np.float32)
            if arr.size >= min_elements_to_quantize and arr.size % 32 == 0:
                orig_tensor_bytes = arr.nbytes
                if target_dtype == DType.Q4_0:
                    q_bytes = quantize_q4_0(arr)
                else:
                    q_bytes = quantize_q8_0(arr)

                orig_bytes += orig_tensor_bytes
                quant_bytes += len(q_bytes)
                tensors_quantized += 1

                if isinstance(graph, GGMLExecutionGraph):
                    target_ggml_type = (
                        GGMLType.GGML_TYPE_Q4_0
                        if target_dtype == DType.Q4_0
                        else GGMLType.GGML_TYPE_Q8_0
                    )
                    q_tensor = GGMLTensorDef(
                        id=tensor.id,
                        name=tensor.name,
                        ggml_type=target_ggml_type,
                        ne=tensor.ne,
                        storage=StorageClass.PARAMETER,
                        data=q_bytes,
                    )
                else:
                    q_tensor = Tensor(
                        id=tensor.id,
                        name=tensor.name,
                        shape=tensor.shape,
                        dtype=target_dtype,
                        storage=StorageClass.PARAMETER,
                        data=q_bytes,
                    )
                new_graph.tensors[tid] = q_tensor
                continue

        # Keep original tensor
        new_graph.tensors[tid] = tensor
        if tensor.data is not None:
            if isinstance(tensor.data, bytes):
                orig_bytes += len(tensor.data)
                quant_bytes += len(tensor.data)
            else:
                nb = np.array(tensor.data).nbytes
                orig_bytes += nb
                quant_bytes += nb

    stats = {
        "tensors_quantized": tensors_quantized,
        "orig_bytes": orig_bytes,
        "quant_bytes": quant_bytes,
        "compression_ratio": (orig_bytes / quant_bytes) if quant_bytes > 0 else 1.0,
    }

    return new_graph, stats
