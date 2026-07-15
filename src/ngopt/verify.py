from __future__ import annotations

from ngopt.ir import Graph, Operation, TensorType

MAX_GRAPH_ELEMENTS = 1_000_000
MAX_GRAPH_COMPUTE = 1_000_000


class VerificationError(ValueError):
    """Raised when a graph violates the supported SSA or tensor semantics."""


def _broadcast_result(lhs: TensorType, rhs: TensorType) -> TensorType | None:
    if lhs == rhs:
        return lhs
    if len(lhs.shape) == 2 and rhs.shape == (lhs.shape[-1],) and lhs.dtype == rhs.dtype:
        return lhs
    if len(rhs.shape) == 2 and lhs.shape == (rhs.shape[-1],) and lhs.dtype == rhs.dtype:
        return rhs
    return None


def _verify_operation(operation: Operation, types: dict[str, TensorType]) -> None:
    operand_types = tuple(types[name] for name in operation.operands)
    if operation.op == "ng.constant":
        if operation.operands:
            raise VerificationError("ng.constant takes no operands")
        return
    expected_arity = {
        "ng.add": 2,
        "ng.mul": 2,
        "ng.matmul": 2,
        "ng.relu": 1,
        "ng.linear_relu": 3,
    }[operation.op]
    if len(operand_types) != expected_arity:
        raise VerificationError(f"{operation.op} expects {expected_arity} operands")
    if operation.op in {"ng.add", "ng.mul"}:
        inferred = _broadcast_result(*operand_types)
        if inferred is None or inferred != operation.result_type:
            raise VerificationError(f"{operation.op} has incompatible tensor shapes")
    elif operation.op == "ng.relu":
        if operand_types[0] != operation.result_type:
            raise VerificationError("ng.relu input and result types must match")
    elif operation.op == "ng.matmul":
        _verify_matmul(operand_types[0], operand_types[1], operation.result_type)
    elif operation.op == "ng.linear_relu":
        _verify_matmul(operand_types[0], operand_types[1], operation.result_type)
        bias = operand_types[2]
        if bias.shape != (operation.result_type.shape[-1],):
            raise VerificationError("ng.linear_relu bias must match the output's last dimension")


def _verify_matmul(lhs: TensorType, rhs: TensorType, result: TensorType) -> None:
    if len(lhs.shape) != 2 or len(rhs.shape) != 2 or len(result.shape) != 2:
        raise VerificationError("ng.matmul requires rank-2 tensors")
    if lhs.shape[1] != rhs.shape[0]:
        raise VerificationError(
            f"ng.matmul inner dimensions must match: {lhs.shape[1]} != {rhs.shape[0]}"
        )
    expected_shape = (lhs.shape[0], rhs.shape[1])
    if result.shape != expected_shape:
        raise VerificationError(
            f"ng.matmul result shape must be {expected_shape}, got {result.shape}"
        )


def estimate_graph_work(graph: Graph) -> int:
    work = 0
    types = {value.name: value.type for value in graph.inputs}
    for operation in graph.operations:
        if operation.op in {"ng.matmul", "ng.linear_relu"}:
            lhs = types[operation.operands[0]]
            rows, shared = lhs.shape
            columns = operation.result_type.shape[1]
            work += rows * shared * columns
        elif operation.op != "ng.constant":
            work += operation.result_type.element_count
        types[operation.result] = operation.result_type
    return work


def verify_graph(graph: Graph) -> Graph:
    if len(graph.outputs) != 1:
        raise VerificationError("graph must return exactly one output")
    types: dict[str, TensorType] = {}
    total_elements = 0
    for value in graph.inputs:
        if value.name in types:
            raise VerificationError(f"duplicate SSA value {value.name}")
        types[value.name] = value.type
        total_elements += value.type.element_count
        if total_elements > MAX_GRAPH_ELEMENTS:
            raise VerificationError(
                f"graph exceeds the {MAX_GRAPH_ELEMENTS:,}-element memory safety limit"
            )
    for operation in graph.operations:
        if operation.result in types:
            raise VerificationError(f"duplicate SSA value {operation.result}")
        for operand in operation.operands:
            if operand not in types:
                raise VerificationError(f"operand {operand} used by {operation.op} is not defined")
        _verify_operation(operation, types)
        types[operation.result] = operation.result_type
        total_elements += operation.result_type.element_count
        if total_elements > MAX_GRAPH_ELEMENTS:
            raise VerificationError(
                f"graph exceeds the {MAX_GRAPH_ELEMENTS:,}-element memory safety limit"
            )
    for output in graph.outputs:
        if output not in types:
            raise VerificationError(f"returned value {output} is not defined")
    if estimate_graph_work(graph) > MAX_GRAPH_COMPUTE:
        raise VerificationError(
            f"graph exceeds the {MAX_GRAPH_COMPUTE:,}-operation compute safety limit"
        )
    return graph
