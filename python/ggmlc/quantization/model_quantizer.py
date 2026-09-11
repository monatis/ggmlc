"""Graph-level neural network parameter quantization with role-based and dynamic policies."""

from __future__ import annotations

from typing import Union
import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.tensor import StorageClass, Tensor
from ggmlc.quantization.policies import QuantizationPolicy, get_quantization_policy
from ggmlc.quantization.quantize import quantize_q4_0, quantize_q8_0
from ggmlc.quantization.roles import TensorRole, classify_tensor_role


def quantize_graph_parameters(
    graph: Graph,
    target_dtype: Union[DType, QuantizationPolicy, str] = DType.Q4_0,
    min_elements_to_quantize: int = 128,
) -> tuple[Graph, dict[str, int | float | dict]]:
    """Quantizes parameter weights in the graph using role-based dynamic quantization.

    Args:
        graph: Source graph (Graph or GGMLExecutionGraph).
        target_dtype: Target precision, QuantizationPolicy, or preset name
                      (e.g., "unsloth_dynamic", "q4_k_m", "q8_0", "q4_0", "f16").
        min_elements_to_quantize: Minimum total elements for a tensor to be quantized.

    Returns:
        (quantized_graph, telemetry_stats)
    """
    policy = get_quantization_policy(target_dtype)

    from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph, GGMLTensorDef
    from ggmlc.dialect.ggml.ops import GGMLType

    if isinstance(graph, GGMLExecutionGraph):
        new_graph = GGMLExecutionGraph(name=f"{graph.name}_{policy.name.lower()}")
        new_graph.symbol_table = list(graph.symbol_table)
    else:
        new_graph = Graph(name=f"{graph.name}_{policy.name.lower()}")
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
    type_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}

    for tid, tensor in graph.tensors.items():
        is_param = tensor.storage == StorageClass.PARAMETER
        tensor_name = getattr(tensor, "name", f"tensor_{tid}") or f"tensor_{tid}"

        # Extract tensor dimensions
        dims: list[int] = []
        if hasattr(tensor, "shape") and hasattr(tensor.shape, "dims"):
            for d in tensor.shape.dims:
                dims.append(int(getattr(d, "value", d)))
        else:
            for d in getattr(tensor, "ne", ()):
                if hasattr(d, "evaluate"):
                    try:
                        dims.append(int(d.evaluate({})))
                    except Exception:  # noqa: BLE001
                        dims.append(1)
                elif hasattr(d, "value"):
                    dims.append(int(d.value))
                else:
                    dims.append(int(d))

        non_unit_dims = [d for d in dims if d > 1]
        is_multi_d = len(non_unit_dims) >= 2

        # 1. Classify architectural role
        role = classify_tensor_role(tensor_name, dims)
        role_counts[role.value] = role_counts.get(role.value, 0) + 1

        # 2. Determine target dtype from policy
        resolved_dtype = policy.resolve_dtype(tensor_name, role, dims)

        has_f32 = (getattr(tensor, "dtype", None) == DType.F32) or (
            getattr(tensor, "ggml_type", None) == GGMLType.GGML_TYPE_F32
        )

        if is_param and tensor.data is not None and has_f32 and is_multi_d and resolved_dtype != DType.F32:
            arr = np.array(tensor.data, dtype=np.float32)
            row_size = (
                dims[0] if isinstance(graph, GGMLExecutionGraph) else (dims[-1] if dims else 1)
            )

            # Block alignment guard for quantized types (Q4_0, Q8_0):
            # Inner dimension must be divisible by 32 and size >= min_elements.
            # If not aligned, fall back safely to F16.
            actual_dtype = resolved_dtype
            if actual_dtype in (DType.Q4_0, DType.Q8_0, DType.Q4_K):
                if arr.size < min_elements_to_quantize or arr.size % 32 != 0 or row_size % 32 != 0 or row_size < 32:
                    actual_dtype = DType.F16

            orig_tensor_bytes = arr.nbytes
            q_bytes: bytes

            if actual_dtype == DType.Q4_0:
                q_bytes = quantize_q4_0(arr)
                target_ggml_type = GGMLType.GGML_TYPE_Q4_0
            elif actual_dtype == DType.Q8_0:
                q_bytes = quantize_q8_0(arr)
                target_ggml_type = GGMLType.GGML_TYPE_Q8_0
            elif actual_dtype in (DType.F16, DType.BF16):
                q_bytes = arr.astype(np.float16).tobytes()
                target_ggml_type = GGMLType.GGML_TYPE_F16
            else:
                q_bytes = arr.tobytes()
                target_ggml_type = GGMLType.GGML_TYPE_F32

            orig_bytes += orig_tensor_bytes
            quant_bytes += len(q_bytes)
            tensors_quantized += 1
            type_name = actual_dtype.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

            if isinstance(graph, GGMLExecutionGraph):
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
                    dtype=actual_dtype,
                    storage=StorageClass.PARAMETER,
                    data=q_bytes,
                )
            new_graph.tensors[tid] = q_tensor
            continue

        # Keep original tensor in F32
        new_graph.tensors[tid] = tensor
        type_counts["F32"] = type_counts.get("F32", 0) + 1
        if tensor.data is not None:
            if isinstance(tensor.data, bytes):
                orig_bytes += len(tensor.data)
                quant_bytes += len(tensor.data)
            else:
                nb = np.array(tensor.data).nbytes
                orig_bytes += nb
                quant_bytes += nb

    stats = {
        "policy": policy.name,
        "tensors_quantized": tensors_quantized,
        "orig_bytes": orig_bytes,
        "quant_bytes": quant_bytes,
        "compression_ratio": (orig_bytes / quant_bytes) if quant_bytes > 0 else 1.0,
        "type_distribution": type_counts,
        "role_distribution": role_counts,
    }

    return new_graph, stats
