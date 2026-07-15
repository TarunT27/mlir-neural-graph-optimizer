from __future__ import annotations

import unittest

from ngopt.ir import Graph, Operation, Tensor, TensorType, Value
from ngopt.passes import default_pipeline
from ngopt.runtime import execute


class RuntimeTests(unittest.TestCase):
    def test_optimized_mlp_matches_original(self) -> None:
        matrix = TensorType((1, 2))
        graph = Graph(
            "main",
            (Value("%x", matrix),),
            (
                Operation.make_constant("%w", Tensor(TensorType((2, 2)), (2.0, -1.0, 0.5, 3.0))),
                Operation.make_constant("%b", Tensor(TensorType((2,)), (0.25, -0.5))),
                Operation("%mm", "ng.matmul", ("%x", "%w"), matrix),
                Operation("%biased", "ng.add", ("%mm", "%b"), matrix),
                Operation("%out", "ng.relu", ("%biased",), matrix),
            ),
            ("%out",),
        )
        inputs = {"%x": Tensor(matrix, (3.0, -2.0))}

        original = execute(graph, inputs)
        optimized_graph = default_pipeline().run(graph).graph
        optimized = execute(optimized_graph, inputs)

        self.assertEqual(original, optimized)
        self.assertEqual(original[0].values, (5.25, 0.0))

    def test_missing_input_is_rejected(self) -> None:
        graph = Graph("main", (Value("%x", TensorType((1,))),), (), ("%x",))

        with self.assertRaisesRegex(ValueError, "missing input %x"):
            execute(graph, {})

    def test_unexpected_input_is_rejected(self) -> None:
        tensor_type = TensorType((1,))
        graph = Graph("main", (Value("%x", tensor_type),), (), ("%x",))

        with self.assertRaisesRegex(ValueError, "unexpected inputs"):
            execute(
                graph,
                {
                    "%x": Tensor(tensor_type, (1.0,)),
                    "%other": Tensor(tensor_type, (2.0,)),
                },
            )

    def test_f32_rounding_is_applied_at_every_operation_boundary(self) -> None:
        tensor_type = TensorType((1,))
        graph = Graph(
            "main",
            (),
            (
                Operation.make_constant("%large", Tensor(tensor_type, (16_777_216.0,))),
                Operation.make_constant("%one", Tensor(tensor_type, (1.0,))),
                Operation("%out", "ng.add", ("%large", "%one"), tensor_type),
            ),
            ("%out",),
        )

        self.assertEqual(execute(graph, {})[0].values, (16_777_216.0,))


if __name__ == "__main__":
    unittest.main()
