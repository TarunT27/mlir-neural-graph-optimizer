from __future__ import annotations

from dataclasses import dataclass
from math import copysign, inf, isfinite, prod
from struct import pack, unpack

SUPPORTED_DTYPES = frozenset({"f32"})
MAX_TENSOR_ELEMENTS = 1_000_000
SUPPORTED_OPERATIONS = frozenset(
    {
        "ng.constant",
        "ng.add",
        "ng.mul",
        "ng.matmul",
        "ng.relu",
        "ng.linear_relu",
    }
)


def to_f32(value: float) -> float:
    """Round a Python number to IEEE-754 binary32."""
    try:
        return float(unpack("!f", pack("!f", float(value)))[0])
    except OverflowError:
        return copysign(inf, value)


@dataclass(frozen=True, slots=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: str = "f32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; supported: f32")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"tensor dimensions must be positive: {self.shape}")
        if len(self.shape) > 2:
            raise ValueError("only scalar, vector, and matrix tensors are supported")
        if (prod(self.shape) if self.shape else 1) > MAX_TENSOR_ELEMENTS:
            raise ValueError(f"tensor exceeds the {MAX_TENSOR_ELEMENTS:,}-element safety limit")

    @property
    def element_count(self) -> int:
        return prod(self.shape) if self.shape else 1

    def __str__(self) -> str:
        dimensions = "x".join(str(dimension) for dimension in self.shape)
        separator = "x" if dimensions else ""
        return f"tensor<{dimensions}{separator}{self.dtype}>"


@dataclass(frozen=True, slots=True)
class Tensor:
    type: TensorType
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = self.type.element_count
        if len(self.values) != expected:
            raise ValueError(
                f"expected {expected} elements for {self.type}, got {len(self.values)}"
            )
        normalized = tuple(to_f32(value) for value in self.values)
        if not all(isfinite(value) for value in normalized):
            raise ValueError("tensor values must be finite f32 numbers")
        object.__setattr__(self, "values", normalized)

    def is_splat(self, value: float) -> bool:
        return all(element == value for element in self.values)


@dataclass(frozen=True, slots=True)
class Value:
    name: str
    type: TensorType

    def __post_init__(self) -> None:
        if not self.name.startswith("%"):
            raise ValueError(f"SSA value must start with %: {self.name!r}")


@dataclass(frozen=True, slots=True)
class Operation:
    result: str
    op: str
    operands: tuple[str, ...]
    result_type: TensorType
    constant: Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))
        if not self.result.startswith("%"):
            raise ValueError(f"operation result must start with %: {self.result!r}")
        if self.op not in SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported operation {self.op!r}")
        if self.op == "ng.constant" and self.constant is None:
            raise ValueError("ng.constant requires a tensor value")
        if self.op != "ng.constant" and self.constant is not None:
            raise ValueError(f"{self.op} cannot carry a constant tensor")
        if self.constant is not None and self.constant.type != self.result_type:
            raise ValueError("constant tensor type must match operation result type")

    @staticmethod
    def make_constant(result: str, tensor: Tensor) -> Operation:
        return Operation(result, "ng.constant", (), tensor.type, tensor)


@dataclass(frozen=True, slots=True)
class Graph:
    name: str
    inputs: tuple[Value, ...]
    operations: tuple[Operation, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "outputs", tuple(self.outputs))

    def type_of(self, value_name: str) -> TensorType:
        for value in self.inputs:
            if value.name == value_name:
                return value.type
        for operation in self.operations:
            if operation.result == value_name:
                return operation.result_type
        raise KeyError(value_name)


@dataclass(frozen=True, slots=True)
class Module:
    name: str
    graph: Graph
