from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any


@unique
class OpCode(Enum):
    # Elementwise arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    NEG = "neg"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    EXP = "exp"
    LOG = "log"
    ABS = "abs"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    SIN = "sin"
    COS = "cos"
    TANH = "tanh"

    # Neural network activations & norms
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    RMS_NORM = "rms_norm"
    LAYER_NORM = "layer_norm"

    # Linear algebra & Attention
    MATMUL = "matmul"
    LINEAR = "linear"
    EMBEDDING = "embedding"
    ROPE = "rope"
    SDPA = "sdpa"

    # Reductions
    SUM = "sum"
    MEAN = "mean"
    AMAX = "amax"
    AMIN = "amin"

    # Tensor manipulation
    RESHAPE = "reshape"
    VIEW = "view"
    PERMUTE = "permute"
    TRANSPOSE = "transpose"
    SLICE = "slice"
    CONCAT = "concat"
    SPLIT = "split"
    EXPAND = "expand"
    SQUEEZE = "squeeze"
    UNSQUEEZE = "unsqueeze"

    # Custom/Special
    SWIGLU = "swiglu"


@dataclass
class Operation:
    """Represents a computational node in Canonical IR."""

    id: int
    opcode: OpCode
    inputs: list[int]
    outputs: list[int]
    attributes: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError("Operation id must be int")
        if not isinstance(self.opcode, OpCode):
            raise TypeError(f"Operation opcode must be OpCode, got {type(self.opcode)}")
        if not isinstance(self.inputs, list):
            self.inputs = list(self.inputs)
        if not isinstance(self.outputs, list):
            self.outputs = list(self.outputs)

    def __repr__(self) -> str:
        attrs = f", attrs={self.attributes}" if self.attributes else ""
        return f"Op(id={self.id}, {self.opcode.name}, in={self.inputs}, out={self.outputs}{attrs})"
