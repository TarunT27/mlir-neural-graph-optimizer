from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ngopt.ir import Graph, Operation, Tensor
from ngopt.runtime import evaluate_operation
from ngopt.verify import verify_graph

PASS_NAMES = (
    "canonicalize",
    "const-fold",
    "fuse-linear-relu",
    "dce",
)


@dataclass(frozen=True, slots=True)
class PassReport:
    passes: tuple[str, ...]
    nodes_before: int
    nodes_after: int
    rewrites: int

    def to_dict(self) -> dict[str, int | list[str]]:
        return {
            "passes": list(self.passes),
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "rewrites": self.rewrites,
        }


@dataclass(frozen=True, slots=True)
class PassResult:
    graph: Graph
    report: PassReport


def _replace_names(graph: Graph, aliases: dict[str, str], kept: list[Operation]) -> Graph:
    def resolve(name: str) -> str:
        while name in aliases:
            name = aliases[name]
        return name

    operations = tuple(
        Operation(
            operation.result,
            operation.op,
            tuple(resolve(operand) for operand in operation.operands),
            operation.result_type,
            operation.constant,
        )
        for operation in kept
    )
    return Graph(
        graph.name, graph.inputs, operations, tuple(resolve(name) for name in graph.outputs)
    )


def _canonicalize(graph: Graph) -> tuple[Graph, int]:
    constants: dict[str, Tensor] = {}
    producers: dict[str, Operation] = {}
    aliases: dict[str, str] = {}
    kept: list[Operation] = []
    rewrites = 0
    types = {
        **{value.name: value.type for value in graph.inputs},
        **{operation.result: operation.result_type for operation in graph.operations},
    }
    for operation in graph.operations:
        operands = tuple(aliases.get(name, name) for name in operation.operands)
        current = Operation(
            operation.result,
            operation.op,
            operands,
            operation.result_type,
            operation.constant,
        )
        if current.constant is not None:
            constants[current.result] = current.constant
        if current.op == "ng.mul":
            left = constants.get(current.operands[0])
            right = constants.get(current.operands[1])
            if (
                left is not None
                and left.is_splat(1.0)
                and types[current.operands[1]] == current.result_type
            ):
                aliases[current.result] = current.operands[1]
                rewrites += 1
                continue
            if (
                right is not None
                and right.is_splat(1.0)
                and types[current.operands[0]] == current.result_type
            ):
                aliases[current.result] = current.operands[0]
                rewrites += 1
                continue
        if current.op == "ng.relu":
            producer = producers.get(current.operands[0])
            if producer is not None and producer.op == "ng.relu":
                aliases[current.result] = current.operands[0]
                rewrites += 1
                continue
        kept.append(current)
        producers[current.result] = current
    return _replace_names(graph, aliases, kept), rewrites


def _const_fold(graph: Graph) -> tuple[Graph, int]:
    constants: dict[str, Tensor] = {}
    operations: list[Operation] = []
    rewrites = 0
    for operation in graph.operations:
        if operation.constant is not None:
            constants[operation.result] = operation.constant
            operations.append(operation)
            continue
        if operation.operands and all(name in constants for name in operation.operands):
            operands = tuple(constants[name] for name in operation.operands)
            folded = evaluate_operation(operation, operands)
            replacement = Operation.make_constant(operation.result, folded)
            constants[operation.result] = folded
            operations.append(replacement)
            rewrites += 1
        else:
            operations.append(operation)
    return Graph(graph.name, graph.inputs, tuple(operations), graph.outputs), rewrites


def _dce(graph: Graph) -> tuple[Graph, int]:
    producers = {operation.result: operation for operation in graph.operations}
    live = set(graph.outputs)
    worklist = list(graph.outputs)
    while worklist:
        name = worklist.pop()
        operation = producers.get(name)
        if operation is None:
            continue
        for operand in operation.operands:
            if operand not in live:
                live.add(operand)
                worklist.append(operand)
    operations = tuple(operation for operation in graph.operations if operation.result in live)
    return Graph(graph.name, graph.inputs, operations, graph.outputs), len(graph.operations) - len(
        operations
    )


def _fuse_linear_relu(graph: Graph) -> tuple[Graph, int]:
    producers = {operation.result: operation for operation in graph.operations}
    types = {
        **{value.name: value.type for value in graph.inputs},
        **{operation.result: operation.result_type for operation in graph.operations},
    }
    uses = Counter(operand for operation in graph.operations for operand in operation.operands)
    uses.update(graph.outputs)
    remove: set[str] = set()
    replacements: dict[str, Operation] = {}
    rewrites = 0
    for relu in graph.operations:
        if relu.op != "ng.relu":
            continue
        add = producers.get(relu.operands[0])
        if add is None or add.op != "ng.add" or uses[add.result] != 1:
            continue
        matmul: Operation | None = None
        bias_name: str | None = None
        for index in (0, 1):
            candidate = producers.get(add.operands[index])
            if candidate is not None and candidate.op == "ng.matmul":
                matmul = candidate
                bias_name = add.operands[1 - index]
                break
        if matmul is None or bias_name is None or uses[matmul.result] != 1:
            continue
        bias_type = types.get(bias_name)
        if bias_type is None:
            continue
        if bias_type.shape != (relu.result_type.shape[-1],):
            continue
        remove.update({matmul.result, add.result})
        replacements[relu.result] = Operation(
            relu.result,
            "ng.linear_relu",
            (*matmul.operands, bias_name),
            relu.result_type,
        )
        rewrites += 1
    operations = tuple(
        replacements.get(operation.result, operation)
        for operation in graph.operations
        if operation.result not in remove
    )
    return Graph(graph.name, graph.inputs, operations, graph.outputs), rewrites


_PASS_FUNCTIONS = {
    "canonicalize": _canonicalize,
    "const-fold": _const_fold,
    "fuse-linear-relu": _fuse_linear_relu,
    "dce": _dce,
}


class PassManager:
    def __init__(self, passes: tuple[str, ...], *, verify_each: bool = True) -> None:
        unknown = tuple(name for name in passes if name not in _PASS_FUNCTIONS)
        if unknown:
            raise ValueError(f"unknown optimizer pass: {', '.join(unknown)}")
        self._passes = tuple(passes)
        self._verify_each = verify_each

    @property
    def passes(self) -> tuple[str, ...]:
        return self._passes

    def run(self, graph: Graph) -> PassResult:
        verify_graph(graph)
        current = graph
        rewrites = 0
        for pass_name in self._passes:
            current, pass_rewrites = _PASS_FUNCTIONS[pass_name](current)
            rewrites += pass_rewrites
            if self._verify_each:
                verify_graph(current)
        report = PassReport(
            self._passes,
            len(graph.operations),
            len(current.operations),
            rewrites,
        )
        return PassResult(current, report)


def default_pipeline() -> PassManager:
    return PassManager(("const-fold", "canonicalize", "fuse-linear-relu", "dce"))
