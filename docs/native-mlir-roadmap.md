# Native MLIR status and roadmap

The first native milestone is complete. [`native_mlir/`](../native_mlir/README.md) is an
out-of-tree MLIR 22.1.6 project, not generated display text.

## Implemented

- TableGen `ng` dialect with `constant`, `add`, `mul`, `matmul`, `relu`, `linear_relu`, and
  `return` operations
- Static rank-0 through rank-2 `f32` verification, positive dimensions, bounded tensor sizes,
  broadcasting checks, MatMul geometry, bias geometry, and function-return checks
- IEEE binary32 constant folding
- Type-safe multiply-by-one and nested-ReLU canonicalization
- One-use guarded MatMul + bias Add + ReLU fusion
- Side-effect-aware dead-code elimination
- Complete conversion to Arith, Tensor, Linalg, and Func with zero-initialized Linalg MatMul
- `mlir-opt`-style native driver and positive/negative FileCheck tests
- Python `native-export` bridge for the dependency-free graph format

## Next stages

1. One-shot bufferization and MemRef lowering.
2. LLVM dialect conversion and a CPU execution runner.
3. Seeded reference-vs-compiled numerical regression.
4. Dynamic-shape design, if a concrete use case justifies the additional verifier surface.
5. ONNX import and broader operator coverage.

Native machine-code generation remains explicitly unsupported until stages 1–3 are complete.
