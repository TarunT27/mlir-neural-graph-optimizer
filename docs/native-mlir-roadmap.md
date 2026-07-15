# Native MLIR integration roadmap

The standalone optimizer is intentionally useful before LLVM is installed. Native integration
will be added as an adapter, not as a second implementation with divergent semantics.

## Planned stages

1. Pin one LLVM release and add a containerized `mlir-opt` toolchain.
2. Define the Neural Graph dialect with TableGen operation and type constraints.
3. Translate standalone NGIR to the native dialect and round-trip it.
4. Port each tested rewrite as an MLIR pattern with positive and negative `lit` tests.
5. Lower supported operations through Linalg/Arith/Tensor and bufferization.
6. Execute compiled CPU output and compare it with the standalone reference runtime.

## Completion bar

Native support will not be marked supported until a clean container build parses every example,
runs the pass suite, eliminates illegal Neural Graph ops during lowering, and passes seeded
original-vs-compiled numerical regression tests.
