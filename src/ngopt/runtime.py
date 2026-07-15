from __future__ import annotations

from collections.abc import Iterable
from math import isclose

from ngopt.ir import Graph, Operation, Tensor, TensorType, to_f32
from ngopt.verify import verify_graph


def _elementwise(lhs: Tensor, rhs: Tensor, operator: str, result_type: TensorType) -> Tensor:
    pairs: Iterable[tuple[float, float]]
    if lhs.type == rhs.type:
        pairs = zip(lhs.values, rhs.values, strict=True)
    elif len(lhs.type.shape) == 2 and len(rhs.type.shape) == 1:
        width = rhs.type.shape[0]
        pairs = ((value, rhs.values[index % width]) for index, value in enumerate(lhs.values))
    elif len(rhs.type.shape) == 2 and len(lhs.type.shape) == 1:
        width = lhs.type.shape[0]
        pairs = ((lhs.values[index % width], value) for index, value in enumerate(rhs.values))
    else:
        raise ValueError(f"cannot broadcast {lhs.type} and {rhs.type}")
    if operator == "ng.add":
        values = tuple(left + right for left, right in pairs)
    else:
        values = tuple(left * right for left, right in pairs)
    return Tensor(result_type, values)


def _matmul(lhs: Tensor, rhs: Tensor, result_type: TensorType) -> Tensor:
    rows, shared = lhs.type.shape
    _, columns = rhs.type.shape
    values_list: list[float] = []
    for row in range(rows):
        for column in range(columns):
            accumulator = 0.0
            for inner in range(shared):
                product = to_f32(
                    lhs.values[row * shared + inner] * rhs.values[inner * columns + column]
                )
                accumulator = to_f32(accumulator + product)
            values_list.append(accumulator)
    values = tuple(values_list)
    return Tensor(result_type, values)


def evaluate_operation(operation: Operation, operands: tuple[Tensor, ...]) -> Tensor:
    if operation.op == "ng.constant":
        assert operation.constant is not None
        return operation.constant
    if operation.op in {"ng.add", "ng.mul"}:
        return _elementwise(operands[0], operands[1], operation.op, operation.result_type)
    if operation.op == "ng.matmul":
        return _matmul(operands[0], operands[1], operation.result_type)
    if operation.op == "ng.relu":
        return Tensor(operation.result_type, tuple(max(value, 0.0) for value in operands[0].values))
    if operation.op == "ng.linear_relu":
        multiplied = _matmul(operands[0], operands[1], operation.result_type)
        biased = _elementwise(multiplied, operands[2], "ng.add", operation.result_type)
        return Tensor(operation.result_type, tuple(max(value, 0.0) for value in biased.values))
    raise ValueError(f"cannot execute unsupported operation {operation.op}")


def execute(graph: Graph, inputs: dict[str, Tensor]) -> tuple[Tensor, ...]:
    verify_graph(graph)
    environment: dict[str, Tensor] = {}
    for value in graph.inputs:
        if value.name not in inputs:
            raise ValueError(f"missing input {value.name}")
        tensor = inputs[value.name]
        if tensor.type != value.type:
            raise ValueError(f"input {value.name} expects {value.type}, got {tensor.type}")
        environment[value.name] = tensor
    unexpected = sorted(set(inputs) - {value.name for value in graph.inputs})
    if unexpected:
        raise ValueError(f"unexpected inputs: {', '.join(unexpected)}")
    for operation in graph.operations:
        operands = tuple(environment[name] for name in operation.operands)
        environment[operation.result] = evaluate_operation(operation, operands)
    return tuple(environment[name] for name in graph.outputs)


def tensors_close(lhs: Tensor, rhs: Tensor, *, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    return lhs.type == rhs.type and all(
        isclose(left, right, rel_tol=rtol, abs_tol=atol)
        for left, right in zip(lhs.values, rhs.values, strict=True)
    )
