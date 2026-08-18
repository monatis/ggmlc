from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Optional

import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.shape import Shape


@unique
class StorageClass(Enum):
    INPUT = "input"
    PARAMETER = "parameter"
    CONSTANT = "constant"
    ACTIVATION = "activation"
    STATE = "state"
    OUTPUT = "output"


@dataclass
class Tensor:
    """Represents a tensor in the Canonical IR."""

    id: int
    name: str
    shape: Shape
    dtype: DType
    storage: StorageClass
    producer_id: Optional[int] = None
    data: Optional[np.ndarray] = None
    role: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError("Tensor id must be int")
        if not isinstance(self.name, str):
            raise TypeError("Tensor name must be str")
        if not isinstance(self.shape, Shape):
            raise TypeError("Tensor shape must be Shape instance")
        if not isinstance(self.dtype, DType):
            raise TypeError("Tensor dtype must be DType instance")
        if not isinstance(self.storage, StorageClass):
            raise TypeError("Tensor storage must be StorageClass instance")

    @property
    def is_constant(self) -> bool:
        return self.storage in (StorageClass.CONSTANT, StorageClass.PARAMETER)

    def size_bytes(self, env: dict[str, int] | None = None) -> int:
        return self.shape.numel(env) * self.dtype.itemsize

    def __repr__(self) -> str:
        return (
            f"Tensor(id={self.id}, name='{self.name}', shape={self.shape}, "
            f"dtype={self.dtype.name}, storage={self.storage.name})"
        )
