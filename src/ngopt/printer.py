from __future__ import annotations

from ngopt.ir import Module, Operation


def _format_float(value: float) -> str:
    text = repr(float(value))
    return text if "." in text or "e" in text.lower() else f"{text}.0"


def _print_operation(operation: Operation) -> str:
    if operation.constant is not None:
        values = ", ".join(_format_float(value) for value in operation.constant.values)
        return f"    {operation.result} = ng.constant dense<[{values}]> : {operation.result_type}"
    operands = ", ".join(operation.operands)
    return f"    {operation.result} = {operation.op} {operands} : {operation.result_type}"


def print_module(module: Module) -> str:
    graph = module.graph
    inputs = ", ".join(f"{value.name}: {value.type}" for value in graph.inputs)
    return_type = graph.type_of(graph.outputs[0])
    lines = [
        f"module @{module.name} {{",
        f"  func.func @{graph.name}({inputs}) -> {return_type} {{",
        *(_print_operation(operation) for operation in graph.operations),
        f"    ng.return {', '.join(graph.outputs)} : {return_type}",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"
