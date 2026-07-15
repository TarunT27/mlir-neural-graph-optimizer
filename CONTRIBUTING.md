# Contributing

Thank you for improving `ng-opt`.

1. Open an issue for semantic changes or new operations.
2. Add a failing test first, including at least one negative legality case for a rewrite.
3. Make the smallest implementation change that passes.
4. Run every quality gate from the README.
5. Update the IR or pass reference when behavior changes.

Passes must return new IR, preserve strict floating-point behavior by default, and remain
idempotent. Do not add silent fallbacks for unsupported syntax, shapes, or dtypes.

Commit messages use `<type>: <description>`, such as `feat: add guarded linear relu fusion`.
