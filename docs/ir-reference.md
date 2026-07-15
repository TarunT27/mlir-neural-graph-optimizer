# Neural Graph IR reference

## Module grammar

```mlir
module @module_name {
  func.func @main(%input: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %constant = ng.constant dense<[1.0, 1.0]> : tensor<2xf32>
    %result = ng.add %input, %constant : tensor<1x2xf32>
    ng.return %result : tensor<1x2xf32>
  }
}
```

The v0.1 format supports one function and one returned tensor. SSA values must be defined before
use and exactly once. Comments begin with `//`.

## Types

- Element type: `f32`
- Rank: zero, one, or two
- Shape: static positive dimensions only
- Examples: `tensor<f32>`, `tensor<4xf32>`, `tensor<2x4xf32>`

## Operations

### `ng.constant`

Creates a dense tensor. The literal element count must match its type.

### `ng.add` and `ng.mul`

Elementwise binary operations. Operands can have identical types, or one operand can be a vector
whose length matches the last dimension of a matrix.

### `ng.matmul`

Rank-two matrix multiplication `(M × K) · (K × N) -> (M × N)`.

### `ng.relu`

Elementwise `max(x, 0)` with unchanged type.

### `ng.linear_relu`

Fused matrix multiplication, vector bias broadcast, and ReLU. Its three operands are input,
weights, and bias. This op is normally produced by the optimizer.

## Diagnostics

Parser diagnostics contain a source and line number. Semantic verifier diagnostics identify the
failing operation or SSA value. Unsupported syntax is an error, not an unregistered-op escape
hatch.
