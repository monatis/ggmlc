from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any


class Dim(ABC):
    """Base class for dimension expressions."""

    @abstractmethod
    def evaluate(self, env: dict[str, int]) -> int:
        """Evaluate dimension given symbol values."""

    @abstractmethod
    def is_static(self) -> bool:
        pass

    @abstractmethod
    def free_symbols(self) -> set[str]:
        pass

    def __add__(self, other: Dim | int) -> Dim:
        return AddDim(self, _to_dim(other))

    def __radd__(self, other: Dim | int) -> Dim:
        return AddDim(_to_dim(other), self)

    def __sub__(self, other: Dim | int) -> Dim:
        return SubDim(self, _to_dim(other))

    def __mul__(self, other: Dim | int) -> Dim:
        return MulDim(self, _to_dim(other))

    def __rmul__(self, other: Dim | int) -> Dim:
        return MulDim(_to_dim(other), self)

    def __floordiv__(self, other: Dim | int) -> Dim:
        return FloorDivDim(self, _to_dim(other))


def _to_dim(val: Dim | int | str) -> Dim:
    if isinstance(val, Dim):
        return val
    if isinstance(val, int):
        return StaticDim(val)
    if isinstance(val, str):
        return SymbolDim(val)
    raise TypeError(f"Cannot convert {type(val)} to Dim")


@dataclass(frozen=True)
class StaticDim(Dim):
    value: int

    def __post_init__(self):
        if not isinstance(self.value, int):
            raise TypeError(f"StaticDim value must be int, got {type(self.value)}")
        if self.value < 0:
            raise ValueError(f"StaticDim value must be non-negative, got {self.value}")

    def evaluate(self, env: dict[str, int]) -> int:
        return self.value

    def is_static(self) -> bool:
        return True

    def free_symbols(self) -> set[str]:
        return set()

    def __repr__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SymbolDim(Dim):
    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"SymbolDim name must be non-empty string, got {self.name}")

    def evaluate(self, env: dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Symbol '{self.name}' not found in environment: {env}")
        val = env[self.name]
        if val <= 0:
            raise ValueError(f"Symbol '{self.name}' must be positive integer, got {val}")
        return val

    def is_static(self) -> bool:
        return False

    def free_symbols(self) -> set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class AddDim(Dim):
    left: Dim
    right: Dim

    def evaluate(self, env: dict[str, int]) -> int:
        return self.left.evaluate(env) + self.right.evaluate(env)

    def is_static(self) -> bool:
        return self.left.is_static() and self.right.is_static()

    def free_symbols(self) -> set[str]:
        return self.left.free_symbols() | self.right.free_symbols()

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass(frozen=True)
class SubDim(Dim):
    left: Dim
    right: Dim

    def evaluate(self, env: dict[str, int]) -> int:
        res = self.left.evaluate(env) - self.right.evaluate(env)
        if res < 0:
            raise ValueError(f"Negative dimension evaluated: {self} with {env} -> {res}")
        return res

    def is_static(self) -> bool:
        return self.left.is_static() and self.right.is_static()

    def free_symbols(self) -> set[str]:
        return self.left.free_symbols() | self.right.free_symbols()

    def __repr__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass(frozen=True)
class MulDim(Dim):
    left: Dim
    right: Dim

    def evaluate(self, env: dict[str, int]) -> int:
        return self.left.evaluate(env) * self.right.evaluate(env)

    def is_static(self) -> bool:
        return self.left.is_static() and self.right.is_static()

    def free_symbols(self) -> set[str]:
        return self.left.free_symbols() | self.right.free_symbols()

    def __repr__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass(frozen=True)
class FloorDivDim(Dim):
    left: Dim
    right: Dim

    def evaluate(self, env: dict[str, int]) -> int:
        r = self.right.evaluate(env)
        if r == 0:
            raise ZeroDivisionError("Division by zero in dimension expression")
        return self.left.evaluate(env) // r

    def is_static(self) -> bool:
        return self.left.is_static() and self.right.is_static()

    def free_symbols(self) -> set[str]:
        return self.left.free_symbols() | self.right.free_symbols()

    def __repr__(self) -> str:
        return f"({self.left} // {self.right})"


@dataclass(frozen=True)
class CeilDivDim(Dim):
    left: Dim
    right: Dim

    def evaluate(self, env: dict[str, int]) -> int:
        r = self.right.evaluate(env)
        if r == 0:
            raise ZeroDivisionError("Division by zero in dimension expression")
        return math.ceil(self.left.evaluate(env) / r)

    def is_static(self) -> bool:
        return self.left.is_static() and self.right.is_static()

    def free_symbols(self) -> set[str]:
        return self.left.free_symbols() | self.right.free_symbols()

    def __repr__(self) -> str:
        return f"ceil({self.left} / {self.right})"


@dataclass(frozen=True)
class Shape:
    """Represents the shape of a tensor with static or symbolic dimensions."""

    dims: tuple[Dim, ...]

    def __init__(self, dims: Sequence[Dim | int | str]):
        object.__setattr__(self, "dims", tuple(_to_dim(d) for d in dims))

    @property
    def rank(self) -> int:
        return len(self.dims)

    def is_static(self) -> bool:
        return all(d.is_static() for d in self.dims)

    def free_symbols(self) -> set[str]:
        res = set()
        for d in self.dims:
            res |= d.free_symbols()
        return res

    def evaluate(self, env: dict[str, int] | None = None) -> tuple[int, ...]:
        env = env or {}
        return tuple(d.evaluate(env) for d in self.dims)

    def numel(self, env: dict[str, int] | None = None) -> int:
        concrete = self.evaluate(env)
        prod = 1
        for d in concrete:
            prod *= d
        return prod

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return Shape(self.dims[idx])
        return self.dims[idx]

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self) -> Iterator[Dim]:
        return iter(self.dims)

    def __repr__(self) -> str:
        return f"Shape([{', '.join(repr(d) for d in self.dims)}])"

    @classmethod
    def from_tuple(cls, dims: Sequence[int]) -> Shape:
        return cls([StaticDim(d) for d in dims])
