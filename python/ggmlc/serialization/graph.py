from __future__ import annotations

import io
import struct

from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph
from ggmlc.ir.shape import (
    AddDim,
    CeilDivDim,
    Dim,
    FloorDivDim,
    MulDim,
    StaticDim,
    SubDim,
    SymbolDim,
)
from ggmlc.ir.tensor import StorageClass

MAGIC = b"GGMLC\x01\x00\x00"
VERSION = 1

# Dim Types
DIM_STATIC = 0
DIM_SYMBOL = 1
DIM_ADD = 2
DIM_SUB = 3
DIM_MUL = 4
DIM_FLOORDIV = 5
DIM_CEILDIV = 6


def _write_str(f: io.BytesIO, s: str) -> None:
    b = s.encode("utf-8")
    f.write(struct.pack("<I", len(b)))
    f.write(b)


def _read_str(f: io.BytesIO) -> str:
    (length,) = struct.unpack("<I", f.read(4))
    b = f.read(length)
    return b.decode("utf-8")


def _serialize_dim(dim: Dim, symbol_map: dict[str, int]) -> bytes:
    if isinstance(dim, StaticDim):
        return struct.pack("<Bq", DIM_STATIC, dim.value)
    elif isinstance(dim, SymbolDim):
        s_idx = symbol_map.get(dim.name, 0)
        return struct.pack("<Bq", DIM_SYMBOL, s_idx)
    elif isinstance(dim, AddDim):
        return (
            struct.pack("<B", DIM_ADD)
            + _serialize_dim(dim.left, symbol_map)
            + _serialize_dim(dim.right, symbol_map)
        )
    elif isinstance(dim, SubDim):
        return (
            struct.pack("<B", DIM_SUB)
            + _serialize_dim(dim.left, symbol_map)
            + _serialize_dim(dim.right, symbol_map)
        )
    elif isinstance(dim, MulDim):
        return (
            struct.pack("<B", DIM_MUL)
            + _serialize_dim(dim.left, symbol_map)
            + _serialize_dim(dim.right, symbol_map)
        )
    elif isinstance(dim, FloorDivDim):
        return (
            struct.pack("<B", DIM_FLOORDIV)
            + _serialize_dim(dim.left, symbol_map)
            + _serialize_dim(dim.right, symbol_map)
        )
    elif isinstance(dim, CeilDivDim):
        return (
            struct.pack("<B", DIM_CEILDIV)
            + _serialize_dim(dim.left, symbol_map)
            + _serialize_dim(dim.right, symbol_map)
        )
    else:
        return struct.pack("<Bq", DIM_STATIC, 1)


def serialize_ggml_graph(graph: GGMLExecutionGraph) -> bytes:
    """Serializes a GGMLExecutionGraph into binary .ggmlc format."""
    f = io.BytesIO()
    data_buffer = io.BytesIO()

    # Header
    f.write(MAGIC)
    f.write(struct.pack("<I", VERSION))
    _write_str(f, graph.name)

    # Symbol Table
    symbol_map = {s: i for i, s in enumerate(graph.symbol_table)}
    f.write(struct.pack("<I", len(graph.symbol_table)))
    for s in graph.symbol_table:
        _write_str(f, s)

    # Graph I/O & Parameters
    f.write(struct.pack("<I", len(graph.inputs)))
    for inp in graph.inputs:
        f.write(struct.pack("<I", inp))

    f.write(struct.pack("<I", len(graph.outputs)))
    for out in graph.outputs:
        f.write(struct.pack("<I", out))

    f.write(struct.pack("<I", len(graph.parameters)))
    for p in graph.parameters:
        f.write(struct.pack("<I", p))

    # Tensors
    f.write(struct.pack("<I", len(graph.tensors)))
    for tid, t in sorted(graph.tensors.items()):
        f.write(struct.pack("<I", t.id))
        _write_str(f, t.name)
        f.write(struct.pack("<i", int(t.ggml_type)))

        # 4 dimensions
        for d in t.ne:
            f.write(_serialize_dim(d, symbol_map))

        # Storage
        storage_map = {
            StorageClass.INPUT: 0,
            StorageClass.PARAMETER: 1,
            StorageClass.CONSTANT: 2,
            StorageClass.ACTIVATION: 3,
            StorageClass.STATE: 4,
            StorageClass.OUTPUT: 5,
        }
        f.write(struct.pack("<i", storage_map[t.storage]))

        # Data payload offset and size
        if t.data is not None:
            offset = data_buffer.tell()
            pad = (16 - (offset % 16)) % 16
            if pad:
                data_buffer.write(b"\x00" * pad)
            offset = data_buffer.tell()

            raw_bytes = t.data.tobytes()
            data_buffer.write(raw_bytes)
            f.write(struct.pack("<QQ", offset, len(raw_bytes)))
        else:
            f.write(struct.pack("<QQ", 0, 0))

    # Operations
    f.write(struct.pack("<I", len(graph.nodes)))
    for op in graph.nodes:
        f.write(struct.pack("<I", op.id))
        f.write(struct.pack("<i", int(op.opcode)))
        _write_str(f, op.name or "")

        f.write(struct.pack("<I", len(op.inputs)))
        for in_id in op.inputs:
            f.write(struct.pack("<I", in_id))

        f.write(struct.pack("<I", len(op.outputs)))
        for out_id in op.outputs:
            f.write(struct.pack("<I", out_id))

        # Serialized attributes (int/float attributes e.g. unary_op, exponent)
        int_attrs = {
            k: round(v) if isinstance(v, float) else int(v)
            for k, v in op.attributes.items()
            if isinstance(v, (int, bool, float))
        }
        f.write(struct.pack("<I", len(int_attrs)))
        for k, v in int_attrs.items():
            _write_str(f, k)
            f.write(struct.pack("<q", v))

    # Append data section
    raw_data = data_buffer.getvalue()
    f.write(struct.pack("<Q", len(raw_data)))
    f.write(raw_data)

    return f.getvalue()
