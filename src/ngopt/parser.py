from __future__ import annotations

import re

from ngopt.ir import Graph, Module, Operation, Tensor, TensorType, Value
from ngopt.verify import VerificationError, verify_graph

_NAME = r"%[A-Za-z_][A-Za-z0-9_.-]*"
_TYPE = r"tensor<[^>]+>"
_MODULE_RE = re.compile(r"module\s+@([A-Za-z_][A-Za-z0-9_.-]*)\s*\{")
_FUNCTION_RE = re.compile(
    rf"func\.func\s+@([A-Za-z_][A-Za-z0-9_.-]*)\((.*)\)\s*->\s*({_TYPE})\s*\{{"
)
_INPUT_RE = re.compile(rf"({_NAME})\s*:\s*({_TYPE})")
_CONSTANT_RE = re.compile(rf"({_NAME})\s*=\s*ng\.constant\s+dense<\[(.*)\]>\s*:\s*({_TYPE})")
_OP_RE = re.compile(rf"({_NAME})\s*=\s*(ng\.[A-Za-z_][A-Za-z0-9_.-]*)\s+(.+?)\s*:\s*({_TYPE})")
_RETURN_RE = re.compile(rf"ng\.return\s+(.+?)\s*:\s*({_TYPE})")


class ParseError(ValueError):
    """A source-aware parser diagnostic."""


def parse_tensor_type(text: str) -> TensorType:
    if not text.startswith("tensor<") or not text.endswith(">"):
        raise ValueError(f"invalid tensor type {text!r}")
    body = text[7:-1]
    components = body.split("x")
    dtype = components[-1]
    try:
        shape = tuple(int(component) for component in components[:-1])
    except ValueError as error:
        raise ValueError(f"invalid static tensor dimensions in {text!r}") from error
    return TensorType(shape, dtype)


def _diagnostic(source: str, line_number: int, message: str) -> ParseError:
    return ParseError(f"{source}:{line_number}: {message}")


def _meaningful_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("//", 1)[0].strip()
        if line:
            lines.append((line_number, line))
    return lines


def parse_module(text: str, *, source: str = "<memory>") -> Module:
    if len(text.encode("utf-8")) > 4 * 1024 * 1024:
        raise _diagnostic(source, 1, "input exceeds the 4 MiB safety limit")
    lines = _meaningful_lines(text)
    if not lines:
        raise _diagnostic(source, 1, "expected 'module @name {'")
    module_match = _MODULE_RE.fullmatch(lines[0][1])
    if module_match is None:
        raise _diagnostic(source, lines[0][0], "expected 'module @name {'")
    if len(lines) < 2:
        raise _diagnostic(source, lines[-1][0], "incomplete module")
    function_match = _FUNCTION_RE.fullmatch(lines[1][1])
    if function_match is None:
        raise _diagnostic(source, lines[1][0], "expected a func.func declaration")
    inputs: list[Value] = []
    input_text = function_match.group(2).strip()
    if input_text:
        for item in input_text.split(","):
            match = _INPUT_RE.fullmatch(item.strip())
            if match is None:
                raise _diagnostic(source, lines[1][0], f"invalid function input {item!r}")
            try:
                inputs.append(Value(match.group(1), parse_tensor_type(match.group(2))))
            except ValueError as error:
                raise _diagnostic(source, lines[1][0], str(error)) from error
    operations: list[Operation] = []
    outputs: tuple[str, ...] | None = None
    return_type: TensorType | None = None
    function_closed = False
    module_closed = False
    for line_number, line in lines[2:]:
        if line == "}":
            if not function_closed:
                function_closed = True
            elif not module_closed:
                module_closed = True
            else:
                raise _diagnostic(source, line_number, "unexpected closing brace")
            continue
        if function_closed:
            raise _diagnostic(source, line_number, "unexpected text after function body")
        if outputs is not None:
            raise _diagnostic(source, line_number, "operation found after ng.return")
        constant_match = _CONSTANT_RE.fullmatch(line)
        if constant_match is not None:
            try:
                tensor_type = parse_tensor_type(constant_match.group(3))
                raw_values = constant_match.group(2).strip()
                values = tuple(
                    float(value.strip()) for value in raw_values.split(",") if value.strip()
                )
                operations.append(
                    Operation.make_constant(constant_match.group(1), Tensor(tensor_type, values))
                )
            except ValueError as error:
                raise _diagnostic(source, line_number, str(error)) from error
            continue
        return_match = _RETURN_RE.fullmatch(line)
        if return_match is not None:
            if outputs is not None:
                raise _diagnostic(source, line_number, "duplicate ng.return")
            outputs = tuple(name.strip() for name in return_match.group(1).split(","))
            try:
                return_type = parse_tensor_type(return_match.group(2))
            except ValueError as error:
                raise _diagnostic(source, line_number, str(error)) from error
            continue
        operation_match = _OP_RE.fullmatch(line)
        if operation_match is not None:
            operation_name = operation_match.group(2)
            operands = tuple(name.strip() for name in operation_match.group(3).split(","))
            try:
                result_type = parse_tensor_type(operation_match.group(4))
                operations.append(
                    Operation(operation_match.group(1), operation_name, operands, result_type)
                )
            except ValueError as error:
                raise _diagnostic(source, line_number, str(error)) from error
            continue
        raise _diagnostic(source, line_number, "unrecognized MLIR-style syntax")
    if not function_closed or not module_closed:
        raise _diagnostic(source, lines[-1][0], "module is missing a closing brace")
    if outputs is None or return_type is None:
        raise _diagnostic(source, lines[-1][0], "function must contain ng.return")
    graph = Graph(function_match.group(1), tuple(inputs), tuple(operations), outputs)
    try:
        verify_graph(graph)
        if len(outputs) != 1 or graph.type_of(outputs[0]) != return_type:
            raise VerificationError("ng.return type does not match the returned value")
        declared_return = parse_tensor_type(function_match.group(3))
        if declared_return != return_type:
            raise VerificationError("function result type does not match ng.return")
    except (VerificationError, ValueError) as error:
        raise _diagnostic(source, lines[-1][0], str(error)) from error
    return Module(module_match.group(1), graph)
