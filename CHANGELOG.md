# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic
versioning.

## [Unreleased]

## [0.2.0] - 2026-07-15

### Added

- Out-of-tree C++17/MLIR 22.1.6 `ng` dialect generated with TableGen
- Native verifiers, IEEE binary32 folding, canonicalization, guarded fusion, and DCE
- Full native conversion to Arith, Tensor, Linalg, and Func dialects
- `ngopt-native-opt` driver with mandatory file, operation, tensor, and compute-work budgets
- Seventeen positive and negative native FileCheck regressions
- `ng-opt native-export` bridge and Python-to-native end-to-end CI validation

### Changed

- Native constant folding now covers broadcast Add/Mul, MatMul, ReLU, and fused LinearReLU
- Native support moved from roadmap-only to a tested optional compiler layer

## [0.1.0] - 2026-07-15

### Added

- Immutable MLIR-style neural graph IR, parser, printer, and verifier
- Constant folding, strict-safe canonicalization, guarded linear/ReLU fusion, and DCE
- Deterministic CPU reference runtime and seeded numerical equivalence validation
- Linalg-oriented lowering text, SVG graph rendering, inspection, and demo commands
- Cross-platform tests, coverage gate, documentation, and repository visuals
