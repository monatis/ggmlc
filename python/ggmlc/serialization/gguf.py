from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ggmlc.dialect.ggml.lowering import (
    GGMLExecutionGraph,
    GGMLOpDef,
    GGMLTensorDef,
)
from ggmlc.dialect.ggml.ops import GGMLOpCode, GGMLType
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

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def _dim_to_dict(dim: Dim) -> dict[str, Any]:
    if isinstance(dim, StaticDim):
        return {"type": "static", "val": dim.value}
    elif isinstance(dim, SymbolDim):
        return {"type": "symbol", "name": dim.name}
    elif isinstance(dim, AddDim):
        return {"type": "add", "left": _dim_to_dict(dim.left), "right": _dim_to_dict(dim.right)}
    elif isinstance(dim, SubDim):
        return {"type": "sub", "left": _dim_to_dict(dim.left), "right": _dim_to_dict(dim.right)}
    elif isinstance(dim, MulDim):
        return {"type": "mul", "left": _dim_to_dict(dim.left), "right": _dim_to_dict(dim.right)}
    elif isinstance(dim, FloorDivDim):
        return {
            "type": "floordiv",
            "left": _dim_to_dict(dim.left),
            "right": _dim_to_dict(dim.right),
        }
    elif isinstance(dim, CeilDivDim):
        return {"type": "ceildiv", "left": _dim_to_dict(dim.left), "right": _dim_to_dict(dim.right)}
    return {"type": "static", "val": 1}


def _dict_to_dim(d: dict[str, Any]) -> Dim:
    t = d.get("type", "static")
    if t == "static":
        return StaticDim(int(d.get("val", 1)))
    elif t == "symbol":
        return SymbolDim(str(d.get("name", "dim")))
    elif t == "add":
        return AddDim(_dict_to_dim(d["left"]), _dict_to_dim(d["right"]))
    elif t == "sub":
        return SubDim(_dict_to_dim(d["left"]), _dict_to_dim(d["right"]))
    elif t == "mul":
        return MulDim(_dict_to_dim(d["left"]), _dict_to_dim(d["right"]))
    elif t == "floordiv":
        return FloorDivDim(_dict_to_dim(d["left"]), _dict_to_dim(d["right"]))
    elif t == "ceildiv":
        return CeilDivDim(_dict_to_dim(d["left"]), _dict_to_dim(d["right"]))
    return StaticDim(1)


STORAGE_TO_INT = {
    StorageClass.INPUT: 0,
    StorageClass.PARAMETER: 1,
    StorageClass.CONSTANT: 2,
    StorageClass.ACTIVATION: 3,
    StorageClass.STATE: 4,
    StorageClass.OUTPUT: 5,
    "input": 0,
    "parameter": 1,
    "constant": 2,
    "activation": 3,
    "state": 4,
    "output": 5,
}

INT_TO_STORAGE = {
    0: StorageClass.INPUT,
    1: StorageClass.PARAMETER,
    2: StorageClass.CONSTANT,
    3: StorageClass.ACTIVATION,
    4: StorageClass.STATE,
    5: StorageClass.OUTPUT,
}


def _serialize_attr_val(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)):
        return v
    elif isinstance(v, StaticDim):
        return v.value
    elif isinstance(v, SymbolDim):
        return v.name
    elif isinstance(v, Dim):
        return _dim_to_dict(v)
    elif isinstance(v, (list, tuple)):
        return [_serialize_attr_val(elem) for elem in v]
    elif isinstance(v, dict):
        return {str(k): _serialize_attr_val(val) for k, val in v.items()}
    return str(v)


def _graph_to_json_spec(graph: GGMLExecutionGraph) -> str:
    """Serializes the execution graph DAG structure into a JSON specification string."""
    tensors_spec = {}
    for tid, t in graph.tensors.items():
        storage_int = STORAGE_TO_INT.get(t.storage, 0)
        tensors_spec[str(tid)] = {
            "id": t.id,
            "name": t.name,
            "type": int(t.ggml_type),
            "ne": [_dim_to_dict(d) for d in t.ne],
            "storage": storage_int,
            "producer_id": t.producer_id,
            "role": t.role,
        }

    nodes_spec = []
    for node in graph.nodes:
        attrs = {k: _serialize_attr_val(v) for k, v in node.attributes.items()}

        nodes_spec.append(
            {
                "id": node.id,
                "opcode": int(node.opcode),
                "inputs": node.inputs,
                "outputs": node.outputs,
                "attributes": attrs,
                "name": node.name or "",
            }
        )

    spec = {
        "name": graph.name,
        "inputs": graph.inputs,
        "outputs": graph.outputs,
        "parameters": graph.parameters,
        "symbol_table": graph.symbol_table,
        "metadata": graph.metadata,
        "tensors": tensors_spec,
        "nodes": nodes_spec,
    }
    return json.dumps(spec, separators=(",", ":"))


def _json_spec_to_graph(spec_str: str) -> GGMLExecutionGraph:
    """Deserializes a JSON specification string back into a GGMLExecutionGraph."""
    spec = json.loads(spec_str)
    g = GGMLExecutionGraph(
        name=spec.get("name", "main"),
        inputs=spec.get("inputs", []),
        outputs=spec.get("outputs", []),
        parameters=spec.get("parameters", []),
        symbol_table=spec.get("symbol_table", []),
        metadata=spec.get("metadata", {}),
    )

    for str_tid, tdata in spec.get("tensors", {}).items():
        tid = int(str_tid)
        ne_dims = tuple(_dict_to_dim(d) for d in tdata.get("ne", []))
        if len(ne_dims) < 4:
            ne_dims = ne_dims + (StaticDim(1),) * (4 - len(ne_dims))

        storage_enum = INT_TO_STORAGE.get(tdata.get("storage", 0), StorageClass.ACTIVATION)

        t = GGMLTensorDef(
            id=tdata["id"],
            name=tdata["name"],
            ggml_type=GGMLType(tdata["type"]),
            ne=ne_dims[:4],
            storage=storage_enum,
            producer_id=tdata.get("producer_id"),
            role=tdata.get("role"),
        )
        g.tensors[tid] = t

    for ndata in spec.get("nodes", []):
        node = GGMLOpDef(
            id=ndata["id"],
            opcode=GGMLOpCode(ndata["opcode"]),
            inputs=ndata["inputs"],
            outputs=ndata["outputs"],
            attributes=ndata.get("attributes", {}),
            name=ndata.get("name"),
        )
        g.nodes.append(node)

    return g


class GGUFWriter:
    """Zero-dependency GGUF v3 Binary Writer."""

    def __init__(self, alignment: int = GGUF_DEFAULT_ALIGNMENT):
        self.alignment = alignment
        self.kv_pairs: list[tuple[str, int, bytes]] = []
        self.tensors: list[dict[str, Any]] = []

    def add_string(self, key: str, val: str) -> None:
        b_val = val.encode("utf-8")
        payload = struct.pack("<Q", len(b_val)) + b_val
        self.kv_pairs.append((key, GGUF_TYPE_STRING, payload))

    def add_uint32(self, key: str, val: int) -> None:
        payload = struct.pack("<I", val)
        self.kv_pairs.append((key, GGUF_TYPE_UINT32, payload))

    def add_int32(self, key: str, val: int) -> None:
        payload = struct.pack("<i", val)
        self.kv_pairs.append((key, GGUF_TYPE_INT32, payload))

    def add_string_array(self, key: str, arr: list[str]) -> None:
        buf = io.BytesIO()
        buf.write(struct.pack("<I", GGUF_TYPE_STRING))
        buf.write(struct.pack("<Q", len(arr)))
        for s in arr:
            b_s = s.encode("utf-8")
            buf.write(struct.pack("<Q", len(b_s)))
            buf.write(b_s)
        self.kv_pairs.append((key, GGUF_TYPE_ARRAY, buf.getvalue()))

    def add_tensor_info(
        self,
        name: str,
        shape: list[int],
        ggml_type: int,
        data: bytes,
    ) -> None:
        self.tensors.append(
            {
                "name": name,
                "shape": shape,
                "type": ggml_type,
                "data": data,
            }
        )

    def write_to_stream(self, out: Any) -> int:
        """Writes the GGUF binary format to an arbitrary binary stream."""
        bytes_written = 0

        # 1. Header
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<I", GGUF_VERSION))
        out.write(struct.pack("<Q", len(self.tensors)))
        out.write(struct.pack("<Q", len(self.kv_pairs)))
        bytes_written += 24

        # 2. Key-Value Metadata Pairs
        for key, val_type, payload in self.kv_pairs:
            b_key = key.encode("utf-8")
            out.write(struct.pack("<Q", len(b_key)))
            out.write(b_key)
            out.write(struct.pack("<I", val_type))
            out.write(payload)
            bytes_written += 8 + len(b_key) + 4 + len(payload)

        # 3. Tensor Info Table (offsets are relative to tensor data payload start)
        # Pre-compute data offsets with 32-byte alignment
        current_offset = 0
        tensor_offsets = []
        for t in self.tensors:
            if current_offset % self.alignment != 0:
                current_offset += self.alignment - (current_offset % self.alignment)
            tensor_offsets.append(current_offset)
            padded_len = ((len(t["data"]) + self.alignment - 1) // self.alignment) * self.alignment
            current_offset += padded_len

        for i, t in enumerate(self.tensors):
            b_name = t["name"].encode("utf-8")
            out.write(struct.pack("<Q", len(b_name)))
            out.write(b_name)
            bytes_written += 8 + len(b_name)

            shape = t["shape"]
            out.write(struct.pack("<I", len(shape)))
            bytes_written += 4
            for d in shape:
                out.write(struct.pack("<Q", d))
                bytes_written += 8

            out.write(struct.pack("<I", t["type"]))
            out.write(struct.pack("<Q", tensor_offsets[i]))
            bytes_written += 4 + 8

        # 4. Align start of tensor data section
        if bytes_written % self.alignment != 0:
            padding = self.alignment - (bytes_written % self.alignment)
            out.write(b"\x00" * padding)
            bytes_written += padding

        # 5. Tensor Data Blobs (streamed directly without large memory copies)
        data_start_pos = bytes_written
        for i, t in enumerate(self.tensors):
            target_pos = data_start_pos + tensor_offsets[i]
            if bytes_written < target_pos:
                gap = target_pos - bytes_written
                out.write(b"\x00" * gap)
                bytes_written += gap

            out.write(t["data"])
            bytes_written += len(t["data"])

            if bytes_written % self.alignment != 0:
                padding = self.alignment - (bytes_written % self.alignment)
                out.write(b"\x00" * padding)
                bytes_written += padding

        return bytes_written

    def write_to_bytes(self) -> bytes:
        """Writes GGUF data to in-memory bytes."""
        out = io.BytesIO()
        self.write_to_stream(out)
        return out.getvalue()

    def write_to_file(self, filepath: str | Path) -> Path:
        """Streams GGUF data directly into a file on disk."""
        p = Path(filepath).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            self.write_to_stream(f)
        return p


def _build_gguf_writer(graph: GGMLExecutionGraph) -> GGUFWriter:
    """Constructs and populates a GGUFWriter with metadata and tensor descriptors."""
    writer = GGUFWriter(alignment=GGUF_DEFAULT_ALIGNMENT)

    # 1. Standard Metadata
    writer.add_string("general.architecture", "ggmlc")
    writer.add_string("general.name", graph.name)
    writer.add_string("ggmlc.version", "1.0.0")
    writer.add_string_array("ggmlc.symbol_table", graph.symbol_table)

    # 2. Graph Spec JSON Metadata
    spec_json = _graph_to_json_spec(graph)
    writer.add_string("ggmlc.graph_spec", spec_json)

    # 3. Add Tensors with data (Parameters / Constants)
    used_names: set[str] = set()
    for _tid, t in sorted(graph.tensors.items()):
        if t.data is not None:
            name = t.name
            if name in used_names:
                name = f"{name}_{_tid}"
            used_names.add(name)

            # Prepare raw byte buffer and match GGUF shape to concrete data
            if isinstance(t.data, (np.ndarray, np.generic, int, float)):
                arr = np.ascontiguousarray(np.asarray(t.data))
                ggml_type_val = int(t.ggml_type)
                if ggml_type_val == int(GGMLType.GGML_TYPE_F32) and arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                elif ggml_type_val == int(GGMLType.GGML_TYPE_I32) and arr.dtype != np.int32:
                    arr = arr.astype(np.int32)
                elif ggml_type_val == int(GGMLType.GGML_TYPE_I64) and arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                elif ggml_type_val == int(GGMLType.GGML_TYPE_F16) and arr.dtype != np.float16:
                    arr = arr.astype(np.float16)
                elif ggml_type_val == int(GGMLType.GGML_TYPE_F64) and arr.dtype != np.float64:
                    arr = arr.astype(np.float64)
                elif ggml_type_val == int(GGMLType.GGML_TYPE_I16) and arr.dtype != np.int16:
                    arr = arr.astype(np.int16)
                elif ggml_type_val == int(GGMLType.GGML_TYPE_I8) and arr.dtype != np.int8:
                    arr = arr.astype(np.int8)
                raw_bytes = arr.tobytes()

                # GGUF tensor shape must match the concrete serialized data shape
                static_shape = list(arr.shape[::-1]) if arr.ndim > 0 else [1]
            elif isinstance(t.data, bytes):
                raw_bytes = t.data
                static_shape = [d.value if isinstance(d, StaticDim) else 1 for d in t.ne]
            else:
                arr = np.ascontiguousarray(np.asarray(t.data))
                raw_bytes = arr.tobytes()
                static_shape = list(arr.shape[::-1]) if arr.ndim > 0 else [1]

            # Ensure 4D
            while len(static_shape) < 4:
                static_shape.append(1)

            writer.add_tensor_info(
                name=name,
                shape=static_shape,
                ggml_type=int(t.ggml_type),
                data=raw_bytes,
            )

    return writer


def serialize_to_gguf(graph: GGMLExecutionGraph) -> bytes:
    """Serializes a GGMLExecutionGraph into official GGUF v3 binary format bytes."""
    writer = _build_gguf_writer(graph)
    return writer.write_to_bytes()


def save_to_gguf(graph: GGMLExecutionGraph, filepath: str | Path) -> Path:
    """Serializes and streams a GGMLExecutionGraph directly to a .gguf file on disk."""
    writer = _build_gguf_writer(graph)
    return writer.write_to_file(filepath)


def deserialize_ggml_graph(data: bytes | Path | str) -> GGMLExecutionGraph:
    """Deserializes GGUF binary or file back into a GGMLExecutionGraph."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()

    if not data.startswith(GGUF_MAGIC):
        raise ValueError("Invalid GGUF binary: magic header mismatch")

    bio = io.BytesIO(data)
    _magic = bio.read(4)
    _version, _n_tensors, n_kv = struct.unpack("<IQQ", bio.read(20))

    graph_spec_str = None
    for _ in range(n_kv):
        key_len = struct.unpack("<Q", bio.read(8))[0]
        key = bio.read(key_len).decode("utf-8")
        val_type = struct.unpack("<I", bio.read(4))[0]

        if val_type == GGUF_TYPE_STRING:
            str_len = struct.unpack("<Q", bio.read(8))[0]
            val_str = bio.read(str_len).decode("utf-8")
            if key == "ggmlc.graph_spec":
                graph_spec_str = val_str
        elif val_type in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
            bio.seek(1, io.SEEK_CUR)
        elif val_type in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
            bio.seek(2, io.SEEK_CUR)
        elif val_type in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
            bio.seek(4, io.SEEK_CUR)
        elif val_type in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
            bio.seek(8, io.SEEK_CUR)
        elif val_type == GGUF_TYPE_ARRAY:
            elem_type, arr_len = struct.unpack("<IQ", bio.read(12))
            if elem_type == GGUF_TYPE_STRING:
                for _ in range(arr_len):
                    s_len = struct.unpack("<Q", bio.read(8))[0]
                    bio.seek(s_len, io.SEEK_CUR)
            elif elem_type in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
                bio.seek(arr_len * 1, io.SEEK_CUR)
            elif elem_type in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
                bio.seek(arr_len * 2, io.SEEK_CUR)
            elif elem_type in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
                bio.seek(arr_len * 4, io.SEEK_CUR)
            elif elem_type in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
                bio.seek(arr_len * 8, io.SEEK_CUR)

    if graph_spec_str:
        return _json_spec_to_graph(graph_spec_str)
    raise ValueError("GGUF metadata missing 'ggmlc.graph_spec'")


def serialize_ggml_graph(graph: GGMLExecutionGraph) -> bytes:
    """Drop-in serialization helper returning GGUF bytes."""
    return serialize_to_gguf(graph)
