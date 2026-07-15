from __future__ import annotations

import argparse
import json
import os
import random
import stat
import sys
import tempfile
from math import isfinite
from pathlib import Path

from ngopt import __version__
from ngopt.ir import Graph, Module, Tensor
from ngopt.lowering import lower_to_linalg
from ngopt.parser import ParseError, parse_module
from ngopt.passes import PASS_NAMES, PassManager, default_pipeline
from ngopt.printer import print_module
from ngopt.runtime import execute, tensors_close
from ngopt.verify import VerificationError, estimate_graph_work
from ngopt.visualize import graph_to_svg

MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_VALIDATION_TRIALS = 1_000
MAX_VALIDATION_WORK = 50_000_000


class CliError(ValueError):
    pass


def _read_source(path_text: str) -> tuple[str, str]:
    if path_text == "-":
        return sys.stdin.read(MAX_INPUT_BYTES + 1), "<stdin>"
    path = Path(path_text)
    try:
        with path.open("rb") as source_file:
            if not stat.S_ISREG(os.fstat(source_file.fileno()).st_mode):
                raise CliError(f"{path}: input must be a regular file")
            data = source_file.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise CliError(f"{path}: input exceeds the 4 MiB safety limit")
        return data.decode("utf-8"), str(path)
    except UnicodeDecodeError as error:
        raise CliError(f"cannot read {path}: input is not valid UTF-8") from error
    except OSError as error:
        raise CliError(f"cannot read {path}: {error.strerror or error}") from error


def _atomic_write(path_text: str | None, content: str) -> None:
    if path_text is None or path_text == "-":
        sys.stdout.write(content)
        return
    path = Path(path_text)
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as error:
        raise CliError(f"cannot write {path}: {error.strerror or error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_module(path_text: str) -> Module:
    text, source = _read_source(path_text)
    return parse_module(text, source=source)


def _manager(pass_text: str | None) -> PassManager:
    if pass_text is None:
        return default_pipeline()
    names = tuple(name.strip() for name in pass_text.split(",") if name.strip())
    if not names:
        raise CliError("the pass pipeline cannot be empty")
    return PassManager(names)


def _parse_inputs(graph: Graph, input_specs: list[str]) -> dict[str, Tensor]:
    raw: dict[str, tuple[float, ...]] = {}
    expected_names = {value.name for value in graph.inputs}
    for spec in input_specs:
        if "=" not in spec:
            raise CliError(f"invalid input {spec!r}; expected %name=v1,v2,...")
        name, values_text = spec.split("=", 1)
        if name not in expected_names:
            raise CliError(f"unknown input {name}")
        if name in raw:
            raise CliError(f"duplicate input {name}")
        try:
            values = tuple(float(value) for value in values_text.split(",") if value)
            if not all(isfinite(value) for value in values):
                raise ValueError("non-finite value")
            raw[name] = values
        except ValueError as error:
            raise CliError(f"invalid numeric value in input {spec!r}") from error
    return {
        value.name: Tensor(value.type, raw[value.name])
        for value in graph.inputs
        if value.name in raw
    }


def _tensor_payload(tensor: Tensor) -> dict[str, str | list[int] | list[float]]:
    return {
        "dtype": tensor.type.dtype,
        "shape": list(tensor.type.shape),
        "values": list(tensor.values),
    }


def _inspect_payload(graph: Graph) -> dict[str, object]:
    parameter_elements = sum(
        operation.result_type.element_count
        for operation in graph.operations
        if operation.constant is not None
    )
    return {
        "function": graph.name,
        "inputs": len(graph.inputs),
        "outputs": len(graph.outputs),
        "nodes": len(graph.operations),
        "parameters": parameter_elements,
        "parameter_bytes": parameter_elements * 4,
        "operations": {
            name: sum(operation.op == name for operation in graph.operations)
            for name in sorted({operation.op for operation in graph.operations})
        },
    }


def _same_destination(first: str | None, second: str | None) -> bool:
    first_sink = first or "-"
    second_sink = second or "-"
    if first_sink == "-" or second_sink == "-":
        return first_sink == second_sink
    return Path(first_sink).resolve() == Path(second_sink).resolve()


def _maybe_optimize(module: Module, *, optimize: bool, passes: str | None) -> Graph:
    if optimize or passes is not None:
        return _manager(passes).run(module.graph).graph
    return module.graph


def _validation_work(original: Graph, optimized: Graph, trials: int) -> int:
    input_elements = sum(value.type.element_count for value in original.inputs)
    output_elements = sum(original.type_of(name).element_count for name in original.outputs)
    per_trial = (
        input_elements
        + estimate_graph_work(original)
        + estimate_graph_work(optimized)
        + 2 * output_elements
    )
    return per_trial * trials


def _cmd_optimize(arguments: argparse.Namespace) -> int:
    module = _load_module(arguments.input)
    if arguments.stats is not None and _same_destination(arguments.output, arguments.stats):
        raise CliError("optimized IR and stats require different destinations")
    result = _manager(arguments.passes).run(module.graph)
    _atomic_write(arguments.output, print_module(Module(module.name, result.graph)))
    if arguments.stats:
        _atomic_write(arguments.stats, json.dumps(result.report.to_dict(), indent=2) + "\n")
    return 0


def _cmd_lower(arguments: argparse.Namespace) -> int:
    module = _load_module(arguments.input)
    graph = _maybe_optimize(module, optimize=arguments.optimize, passes=arguments.passes)
    _atomic_write(arguments.output, lower_to_linalg(graph))
    return 0


def _cmd_run(arguments: argparse.Namespace) -> int:
    module = _load_module(arguments.input)
    graph = _maybe_optimize(module, optimize=arguments.optimize, passes=arguments.passes)
    tensors = _parse_inputs(graph, arguments.input_value)
    outputs = execute(graph, tensors)
    sys.stdout.write(
        json.dumps(
            {"outputs": [_tensor_payload(tensor) for tensor in outputs]},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


def _cmd_inspect(arguments: argparse.Namespace) -> int:
    module = _load_module(arguments.input)
    sys.stdout.write(json.dumps(_inspect_payload(module.graph), indent=2) + "\n")
    return 0


def _cmd_validate(arguments: argparse.Namespace) -> int:
    module = _load_module(arguments.input)
    optimized = _manager(arguments.passes).run(module.graph).graph
    validation_work = _validation_work(module.graph, optimized, arguments.trials)
    if validation_work > MAX_VALIDATION_WORK:
        raise CliError(f"validation exceeds the {MAX_VALIDATION_WORK:,}-operation safety budget")
    randomizer = random.Random(arguments.seed)
    maximum_error = 0.0
    for _ in range(arguments.trials):
        inputs = {
            value.name: Tensor(
                value.type,
                tuple(randomizer.uniform(-2.0, 2.0) for _ in range(value.type.element_count)),
            )
            for value in module.graph.inputs
        }
        original_outputs = execute(module.graph, inputs)
        optimized_outputs = execute(optimized, inputs)
        for original, candidate in zip(original_outputs, optimized_outputs, strict=True):
            maximum_error = max(
                maximum_error,
                *(
                    abs(left - right)
                    for left, right in zip(original.values, candidate.values, strict=True)
                ),
            )
            if not tensors_close(original, candidate, rtol=arguments.rtol, atol=arguments.atol):
                raise CliError(f"numerical mismatch detected with seed {arguments.seed}")
    payload = {
        "equivalent": True,
        "trials": arguments.trials,
        "seed": arguments.seed,
        "rtol": arguments.rtol,
        "atol": arguments.atol,
        "max_abs_error": maximum_error,
    }
    sys.stdout.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return 0


def _cmd_visualize(arguments: argparse.Namespace) -> int:
    module = _load_module(arguments.input)
    graph = _maybe_optimize(module, optimize=arguments.optimize, passes=arguments.passes)
    _atomic_write(arguments.output, graph_to_svg(graph))
    return 0


def _cmd_demo(arguments: argparse.Namespace) -> int:
    module = _load_module(arguments.input)
    result = default_pipeline().run(module.graph)
    output_directory = Path(arguments.output_dir)
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CliError(
            f"cannot create demo directory {output_directory}: {error.strerror or error}"
        ) from error
    _atomic_write(str(output_directory / "before.svg"), graph_to_svg(module.graph))
    _atomic_write(str(output_directory / "after.svg"), graph_to_svg(result.graph))
    _atomic_write(
        str(output_directory / "optimized.mlir"), print_module(Module(module.name, result.graph))
    )
    _atomic_write(str(output_directory / "lowered.mlir"), lower_to_linalg(result.graph))
    _atomic_write(
        str(output_directory / "stats.json"), json.dumps(result.report.to_dict(), indent=2) + "\n"
    )
    validate_arguments = argparse.Namespace(
        input=arguments.input,
        passes=None,
        seed=42,
        trials=20,
        rtol=1e-5,
        atol=1e-6,
    )
    return _cmd_validate(validate_arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ng-opt",
        description="MLIR-style neural graph optimizer with a deterministic CPU reference runtime.",
    )
    parser.add_argument("--version", action="version", version=f"ng-opt {__version__}")
    parser.add_argument(
        "--list-passes", action="store_true", help="list registered optimizer passes"
    )
    subparsers = parser.add_subparsers(dest="command")

    optimize = subparsers.add_parser("optimize", help="optimize MLIR-style neural graph text")
    optimize.add_argument("input", help="input .mlir file, or - for stdin")
    optimize.add_argument("-o", "--output", help="output file; stdout when omitted")
    optimize.add_argument("--passes", help="comma-separated pass pipeline")
    optimize.add_argument("--stats", help="write optimization statistics as JSON")
    optimize.set_defaults(handler=_cmd_optimize)

    lower = subparsers.add_parser(
        "lower", help="lower supported neural ops to linalg-oriented text"
    )
    lower.add_argument("input")
    lower.add_argument("-o", "--output")
    lower.add_argument("--optimize", action="store_true")
    lower.add_argument("--passes")
    lower.set_defaults(handler=_cmd_lower)

    run = subparsers.add_parser("run", help="execute a graph on the reference CPU runtime")
    run.add_argument("input")
    run.add_argument(
        "--input", dest="input_value", action="append", default=[], help="%%name=v1,v2,..."
    )
    run.add_argument("--optimize", action="store_true")
    run.add_argument("--passes")
    run.set_defaults(handler=_cmd_run)

    inspect = subparsers.add_parser("inspect", help="report graph structure and parameter counts")
    inspect.add_argument("input")
    inspect.set_defaults(handler=_cmd_inspect)

    validate = subparsers.add_parser("validate", help="compare original and optimized outputs")
    validate.add_argument("input")
    validate.add_argument("--passes")
    validate.add_argument("--trials", type=int, default=20)
    validate.add_argument("--seed", type=int, default=42)
    validate.add_argument("--rtol", type=float, default=1e-5)
    validate.add_argument("--atol", type=float, default=1e-6)
    validate.set_defaults(handler=_cmd_validate)

    visualize = subparsers.add_parser("visualize", help="render a graph as a standalone SVG")
    visualize.add_argument("input")
    visualize.add_argument("-o", "--output", required=True)
    visualize.add_argument("--optimize", action="store_true")
    visualize.add_argument("--passes")
    visualize.set_defaults(handler=_cmd_visualize)

    demo = subparsers.add_parser("demo", help="generate an end-to-end optimization artifact bundle")
    demo.add_argument("input", nargs="?", default="examples/tiny_mlp.mlir")
    demo.add_argument("--output-dir", default="artifacts/demo")
    demo.set_defaults(handler=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.list_passes:
        sys.stdout.write("\n".join(PASS_NAMES) + "\n")
        return 0
    if not hasattr(arguments, "handler"):
        parser.print_help()
        return 0
    try:
        trials = getattr(arguments, "trials", 1)
        if trials <= 0:
            raise CliError("--trials must be positive")
        if trials > MAX_VALIDATION_TRIALS:
            raise CliError(f"--trials cannot exceed {MAX_VALIDATION_TRIALS}")
        rtol = getattr(arguments, "rtol", 0.0)
        atol = getattr(arguments, "atol", 0.0)
        if not isfinite(rtol) or not isfinite(atol) or rtol < 0 or atol < 0:
            raise CliError("tolerances must be finite and non-negative")
        return int(arguments.handler(arguments))
    except (CliError, ParseError, VerificationError, ValueError) as error:
        sys.stderr.write(f"ng-opt: error: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
