# Optimization passes

## `const-fold`

Walks operations in SSA order and evaluates a pure operation when every operand is a known
constant. Chained constants fold in a single traversal because newly created constants enter the
environment immediately.

## `canonicalize`

Implements only strict-safe, shape-preserving rules:

- `x * splat(1.0) -> x`
- `splat(1.0) * x -> x`
- `relu(relu(x)) -> relu(x)`

It deliberately does not rewrite `x + 0`, `x * 0`, reassociate expressions, or introduce FMA;
those can change signed-zero, NaN, or rounding behavior.

## `fuse-linear-relu`

Matches `matmul -> bias add -> relu` and creates `ng.linear_relu` only when:

- MatMul and Add intermediates each have exactly one use,
- the bias is a vector,
- bias length equals the output's last dimension, and
- normal verifier rules already hold.

The pass preserves the final SSA name and result type.

## `dce`

Computes reverse reachability from graph returns and removes other pure operations. Every current
operation is pure, so there is no side-effect exception in v0.1.

## Default pipeline

`const-fold,canonicalize,fuse-linear-relu,dce`

The pipeline is deterministic and idempotent on the supported IR.
