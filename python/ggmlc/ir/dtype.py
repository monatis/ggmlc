from __future__ import annotations

from enum import Enum, unique
from typing import Any


@unique
class DType(Enum):
    """Supported data types in ggmlc canonical IR."""

    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I32 = "i32"
    I64 = "i64"
    I8 = "i8"
    BOOL = "bool"
    Q4_0 = "q4_0"
    Q4_K = "q4_k"
    Q8_0 = "q8_0"

    @property
    def itemsize(self) -> int | float:
        """Returns size in bytes per element."""
        sizes = {
            DType.F32: 4,
            DType.F16: 2,
            DType.BF16: 2,
            DType.I32: 4,
            DType.I64: 8,
            DType.I8: 1,
            DType.BOOL: 1,
            DType.Q4_0: 0.5625,  # 18 bytes per 32 elements
            DType.Q4_K: 0.5625,
            DType.Q8_0: 1.0625,  # 34 bytes per 32 elements
        }
        return sizes[self]

    @property
    def is_quantized(self) -> bool:
        return self in (DType.Q4_0, DType.Q4_K, DType.Q8_0)

    @property
    def is_floating_point(self) -> bool:
        return self in (DType.F32, DType.F16, DType.BF16)

    @classmethod
    def from_torch(cls, torch_dtype: Any) -> DType:
        """Map PyTorch dtype to DType."""
        import torch

        mapping = {
            torch.float32: cls.F32,
            torch.float16: cls.F16,
            torch.bfloat16: cls.BF16,
            torch.int32: cls.I32,
            torch.int64: cls.I64,
            torch.int8: cls.I8,
            torch.bool: cls.BOOL,
        }
        if torch_dtype not in mapping:
            raise TypeError(f"Unsupported PyTorch dtype: {torch_dtype}")
        return mapping[torch_dtype]

    def to_torch(self) -> Any:
        """Map DType to PyTorch dtype."""
        import torch

        return {
            DType.F32: torch.float32,
            DType.F16: torch.float16,
            DType.I32: torch.int32,
            DType.I16: torch.int16,
            DType.I8: torch.int8,
            DType.BOOL: torch.bool,
        }[self]

    @classmethod
    def from_numpy(cls, np_dtype: Any) -> DType:
        import numpy as np

        np_dt = np.dtype(np_dtype)
        mapping = {
            np.dtype(np.float32): cls.F32,
            np.dtype(np.float16): cls.F16,
            np.dtype(np.int32): cls.I32,
            np.dtype(np.int64): cls.I64,
            np.dtype(np.int8): cls.I8,
            np.dtype(np.bool_): cls.BOOL,
        }
        if np_dt not in mapping:
            raise TypeError(f"Unsupported NumPy dtype: {np_dtype}")
        return mapping[np_dt]

    @classmethod
    def from_str(cls, s: str) -> DType:
        s_lower = s.lower().strip()
        for member in cls:
            if member.value == s_lower or member.name.lower() == s_lower:
                return member
        raise ValueError(f"Unknown DType string: '{s}'")
