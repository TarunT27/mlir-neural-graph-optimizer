# Native C++/MLIR optimizer

This optional layer turns the project into a real out-of-tree MLIR compiler component. It is
pinned to LLVM/MLIR **22.1.6** and keeps the dependency-free Python optimizer as the behavioral
reference for their shared verified tensor surface. Each layer enforces its own resource ceilings.

## Prerequisites

- LLVM/MLIR 22.1.6 built with `-DLLVM_ENABLE_PROJECTS=mlir`
- CMake 3.20 or newer, Ninja, and a C++17 compiler
- Python 3 for the small cross-platform native test runner

The verified LLVM 22.1.6 source archive is
`llvm-project-22.1.6.src.tar.xz` with SHA-256
`6e0b376a1f6d9873e7dfb09ae6e04b9c7024400f01733fa4c29be69d5c138bc2`.

## Build and test

```bash
cmake -S native_mlir -B native_mlir/build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build native_mlir/build --target ngopt-native-opt
cmake --build native_mlir/build --target check-ng
```

`check-ng` runs every `.mlir` fixture through `ngopt-native-opt` and LLVM `FileCheck`, including
invalid-shape and fusion-negative cases.

## End-to-end example

```bash
ng-opt native-export examples/tiny_mlp.mlir --optimize -o native.mlir
native_mlir/build/bin/ngopt-native-opt \
  --ng-constant-fold \
  --ng-canonicalize \
  --ng-fuse-linear-relu \
  --ng-dce \
  --convert-ng-to-linalg \
  native.mlir -o lowered.mlir
```

On Windows, append `.exe` to the native driver name.
The native driver accepts one regular input file up to 4 MiB; stdin is intentionally disabled.
An always-on guard also limits operation count, cumulative tensor elements, and estimated
compute work before user passes.

## Native passes

| Flag | Result |
|---|---|
| `--ng-constant-fold` | Folds dense constants using `APFloat` binary32 semantics |
| `--ng-canonicalize` | Applies strict-safe graph identities |
| `--ng-fuse-linear-relu` | Fuses one-use MatMul + bias Add + ReLU chains |
| `--ng-dce` | Erases unused pure NG operations |
| `--convert-ng-to-linalg` | Eliminates every NG op in favor of standard MLIR dialects |

The lowering target is structured MLIR, not machine code. Bufferization, LLVM conversion, and
compiled CPU numerical regression are tracked in the [native roadmap](../docs/native-mlir-roadmap.md).
