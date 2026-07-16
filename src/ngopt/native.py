from __future__ import annotations

from collections.abc import Mapping

from ngopt.ir import Module, Operation, Tensor, TensorType


def _format_float(value: float) -> str:
    text = repr(float(value))
    return text if "." in text or "e" in text.lower() else f"{text}.0"


def _format_dense(tensor: Tensor) -> str:
    shape = tensor.type.shape
    if not shape:
        return _format_float(tensor.values[0])
    if len(shape) == 1:
        values = ", ".join(_format_float(value) for value in tensor.values)
        return f"[{values}]"
    columns = shape[1]
    rows = tuple(
        tensor.values[index : index + columns] for index in range(0, len(tensor.values), columns)
    )
    formatted_rows = ", ".join(
        "[" + ", ".join(_format_float(value) for value in row) + "]" for row in rows
    )
    return f"[{formatted_rows}]"


def _print_native_operation(value_types: Mapping[str, TensorType], operation: Operation) -> str:
    if operation.constant is not None:
        dense = _format_dense(operation.constant)
        return f"    {operation.result} = ng.constant dense<{dense}> : {operation.result_type}"

    operands = ", ".join(operation.operands)
    if operation.op == "ng.relu":
        signature = str(operation.result_type)
    else:
        operand_types = ", ".join(str(value_types[name]) for name in operation.operands)
        signature = f"({operand_types}) -> {operation.result_type}"
    return f"    {operation.result} = {operation.op} {operands} : {signature}"


def print_native_module(module: Module) -> str:
    """Render the Python graph in syntax accepted by the native NG dialect."""
    graph = module.graph
    inputs = ", ".join(f"{value.name}: {value.type}" for value in graph.inputs)
    return_type = graph.type_of(graph.outputs[0])
    value_types = {value.name: value.type for value in graph.inputs} | {
        operation.result: operation.result_type for operation in graph.operations
    }
    operations = tuple(
        _print_native_operation(value_types, operation) for operation in graph.operations
    )
    lines = (
        f"module @{module.name} {{",
        f"  func.func @{graph.name}({inputs}) -> {return_type} {{",
        *operations,
        f"    ng.return {graph.outputs[0]} : {return_type}",
        "  }",
        "}",
    )
    return "\n".join(lines) + "\n"
