from __future__ import annotations

from ngopt.ir import Graph, Operation
from ngopt.printer import _format_float
from ngopt.verify import verify_graph


def _lower_operation(operation: Operation) -> str:
    if operation.constant is not None:
        values = ", ".join(_format_float(value) for value in operation.constant.values)
        return (
            f"    {operation.result} = arith.constant dense<[{values}]> : {operation.result_type}"
        )
    operands = ", ".join(operation.operands)
    if operation.op == "ng.matmul":
        return f"    {operation.result} = linalg.matmul ins({operands}) : {operation.result_type}"
    if operation.op in {"ng.add", "ng.mul"}:
        function = operation.op.removeprefix("ng.")
        return (
            f'    {operation.result} = linalg.elemwise_binary {{fun = "{function}"}} '
            f"ins({operands}) : {operation.result_type}"
        )
    if operation.op == "ng.relu":
        return (
            f'    {operation.result} = linalg.map {{fun = "relu"}} '
            f"ins({operands}) : {operation.result_type}"
        )
    if operation.op == "ng.linear_relu":
        return (
            f"    {operation.result} = linalg.generic "
            f'{{library_call = "ngopt_gemm_bias_relu"}} ins({operands}) : {operation.result_type}'
        )
    raise ValueError(f"cannot lower unsupported operation {operation.op}")


def lower_to_linalg(graph: Graph) -> str:
    """Render the supported graph as readable standard-dialect-oriented MLIR text.

    This emitter is intentionally a bridge artifact, not a claim that LLVM's
    verifier has validated the text. The standalone interpreter is the executable
    reference backend.
    """
    verify_graph(graph)
    inputs = ", ".join(f"{value.name}: {value.type}" for value in graph.inputs)
    return_type = graph.type_of(graph.outputs[0])
    lines = [
        f"module @{graph.name}_lowered {{",
        f"  func.func @{graph.name}({inputs}) -> {return_type} {{",
        *(_lower_operation(operation) for operation in graph.operations),
        f"    func.return {', '.join(graph.outputs)} : {return_type}",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"
