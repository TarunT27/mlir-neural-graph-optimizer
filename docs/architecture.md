# Architecture

`ng-opt` is a compact compiler pipeline with one semantic source of truth. Text is parsed into
an immutable graph, verified, rewritten by pure pass functions, and then either executed by the
reference runtime or rendered as Linalg-oriented bridge text.

## Components

| Module | Responsibility |
|---|---|
| `ir.py` | Frozen tensor, value, operation, graph, and module records |
| `parser.py` / `printer.py` | Source-aware parser and stable canonical printer |
| `verify.py` | SSA, arity, type, shape, broadcast, and return invariants |
| `passes.py` | Pass manager, immutable rewrites, reports, and default pipeline |
| `runtime.py` | Deterministic reference kernels and graph execution |
| `lowering.py` | Readable Linalg-oriented bridge text |
| `visualize.py` | Deterministic, escaped, dependency-free SVG graphs |
| `cli.py` | File/stdin boundaries, atomic outputs, validation, and demo orchestration |

## Data flow

1. The parser accepts a deliberately small MLIR-style grammar.
2. The verifier rejects unknown values, duplicate SSA definitions, illegal ranks, bad MatMul
   shapes, invalid broadcasts, and mismatched return types.
3. The pass manager runs registered rewrites and re-verifies after every pass.
4. The interpreter executes original or optimized IR using the same operation semantics used
   by constant folding.
5. Numerical validation generates deterministic inputs and compares observable outputs.

Using the runtime's operation evaluator during constant folding prevents the compiler and
runtime from drifting into subtly different semantics.

## Immutability contract

Public IR records are frozen dataclasses and use tuples for all collections. A pass constructs a
new `Graph`; it never edits an operation owned by the caller. Derived producer/use maps exist
only inside a pass invocation.

## Trust boundaries

- Text input is untrusted and capped at 4 MiB.
- Static shapes must be positive and rank two or less.
- Tensor shapes are capped at 1,000,000 elements before allocation or random validation.
- Cumulative graph elements and estimated scalar work are capped at 1,000,000 each.
- Validation is capped by both trial count and total estimated work.
- Tensor values and comparison tolerances must be finite.
- Tensor literals must contain exactly the declared number of values.
- Output writes use a sibling temporary file followed by replacement.
- SVG labels are HTML-escaped.
- CLI errors are concise and do not expose stack traces during normal failures.

## Native MLIR boundary

The project uses MLIR's textual and SSA design ideas, but its parser is not LLVM's parser and the
emitted lowering text is not advertised as `mlir-opt`-verified. The staged integration plan is in
[native-mlir-roadmap.md](native-mlir-roadmap.md).
