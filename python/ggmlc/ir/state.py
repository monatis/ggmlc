from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.shape import Shape


@dataclass
class StateDeclaration:
    """Declares a persistent mutable state tensor (e.g. KV cache, RNN state)."""

    id: int
    name: str
    shape: Shape
    dtype: DType
    initial_data: np.ndarray | None = None
    role: str | None = None

    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError("State id must be int")
        if not isinstance(self.name, str):
            raise TypeError("State name must be str")
        if not isinstance(self.shape, Shape):
            raise TypeError("State shape must be Shape instance")
        if not isinstance(self.dtype, DType):
            raise TypeError("State dtype must be DType instance")
