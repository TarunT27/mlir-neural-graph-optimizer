from __future__ import annotations

import unittest

from ngopt.ir import Graph, Operation, Tensor, TensorType, Value
from ngopt.passes import PassManager, default_pipeline


class OptimizerPassTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = TensorType((2,))

    def test_constant_folding_reaches_a_fixed_point(self) -> None:
        graph = Graph(
            "main",
            (),
            (
                Operation.make_constant("%a", Tensor(self.vector, (1.0, -3.0))),
                Operation.make_constant("%b", Tensor(self.vector, (2.0, 4.0))),
                Operation("%sum", "ng.add", ("%a", "%b"), self.vector),
                Operation("%out", "ng.relu", ("%sum",), self.vector),
            ),
            ("%out",),
        )

        result = PassManager(("const-fold", "dce")).run(graph)

        self.assertEqual(len(result.graph.operations), 1)
        self.assertEqual(result.graph.operations[0].op, "ng.constant")
        self.assertEqual(result.graph.operations[0].constant.values, (3.0, 1.0))
        self.assertGreaterEqual(result.report.rewrites, 2)

    def test_mul_by_one_is_simplified_without_mutating_input(self) -> None:
        graph = Graph(
            "main",
            (Value("%x", self.vector),),
            (
                Operation.make_constant("%one", Tensor(self.vector, (1.0, 1.0))),
                Operation("%out", "ng.mul", ("%x", "%one"), self.vector),
            ),
            ("%out",),
        )

        result = PassManager(("canonicalize", "dce")).run(graph)

        self.assertEqual(graph.outputs, ("%out",))
        self.assertEqual(result.graph.outputs, ("%x",))
        self.assertEqual(result.graph.operations, ())

    def test_dead_chain_is_removed(self) -> None:
        graph = Graph(
            "main",
            (Value("%x", self.vector),),
            (
                Operation("%live", "ng.relu", ("%x",), self.vector),
                Operation("%dead1", "ng.relu", ("%x",), self.vector),
                Operation("%dead2", "ng.relu", ("%dead1",), self.vector),
            ),
            ("%live",),
        )

        result = PassManager(("dce",)).run(graph)

        self.assertEqual(tuple(op.result for op in result.graph.operations), ("%live",))

    def test_linear_relu_fuses_only_for_single_use_intermediate(self) -> None:
        matrix = TensorType((1, 2))
        weights = TensorType((2, 2))
        graph = Graph(
            "main",
            (Value("%x", matrix),),
            (
                Operation.make_constant("%w", Tensor(weights, (1.0, 0.0, 0.0, 1.0))),
                Operation.make_constant("%b", Tensor(TensorType((2,)), (0.5, -0.5))),
                Operation("%mm", "ng.matmul", ("%x", "%w"), matrix),
                Operation("%biased", "ng.add", ("%mm", "%b"), matrix),
                Operation("%out", "ng.relu", ("%biased",), matrix),
            ),
            ("%out",),
        )

        result = PassManager(("fuse-linear-relu", "dce")).run(graph)

        operations = {op.result: op for op in result.graph.operations}
        self.assertEqual(operations["%out"].op, "ng.linear_relu")
        self.assertEqual(operations["%out"].operands, ("%x", "%w", "%b"))

    def test_default_pipeline_is_idempotent(self) -> None:
        graph = Graph(
            "main",
            (Value("%x", self.vector),),
            (
                Operation.make_constant("%one", Tensor(self.vector, (1.0, 1.0))),
                Operation("%mul", "ng.mul", ("%x", "%one"), self.vector),
                Operation("%out", "ng.relu", ("%mul",), self.vector),
            ),
            ("%out",),
        )

        once = default_pipeline().run(graph).graph
        twice = default_pipeline().run(once).graph

        self.assertEqual(once, twice)

    def test_fusion_is_blocked_when_matmul_has_multiple_uses(self) -> None:
        matrix = TensorType((1, 2))
        graph = Graph(
            "main",
            (Value("%x", matrix),),
            (
                Operation.make_constant("%w", Tensor(TensorType((2, 2)), (1.0, 0.0, 0.0, 1.0))),
                Operation.make_constant("%b", Tensor(TensorType((2,)), (0.0, 0.0))),
                Operation("%mm", "ng.matmul", ("%x", "%w"), matrix),
                Operation("%biased", "ng.add", ("%mm", "%b"), matrix),
                Operation("%activated", "ng.relu", ("%biased",), matrix),
                Operation("%out", "ng.add", ("%activated", "%mm"), matrix),
            ),
            ("%out",),
        )

        result = PassManager(("fuse-linear-relu",)).run(graph)

        self.assertNotIn("ng.linear_relu", {operation.op for operation in result.graph.operations})

    def test_unknown_pass_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown optimizer pass"):
            PassManager(("quantize-magic",))

    def test_broadcasted_one_is_not_replaced_with_wrong_shape(self) -> None:
        vector = TensorType((2,))
        matrix = TensorType((2, 2))
        graph = Graph(
            "main",
            (Value("%x", vector),),
            (
                Operation.make_constant("%ones", Tensor(matrix, (1.0, 1.0, 1.0, 1.0))),
                Operation("%out", "ng.mul", ("%ones", "%x"), matrix),
            ),
            ("%out",),
        )

        result = PassManager(("canonicalize",)).run(graph)

        self.assertEqual(result.graph.outputs, ("%out",))
        self.assertEqual(result.graph.type_of("%out"), matrix)

    def test_broadcasted_vector_one_simplifies_for_both_operand_orders(self) -> None:
        vector = TensorType((2,))
        matrix = TensorType((2, 2))
        for operands in (("%x", "%ones"), ("%ones", "%x")):
            with self.subTest(operands=operands):
                graph = Graph(
                    "main",
                    (Value("%x", matrix),),
                    (
                        Operation.make_constant("%ones", Tensor(vector, (1.0, 1.0))),
                        Operation("%out", "ng.mul", operands, matrix),
                    ),
                    ("%out",),
                )

                result = PassManager(("canonicalize", "dce")).run(graph)

                self.assertEqual(result.graph.outputs, ("%x",))
                self.assertEqual(result.graph.operations, ())


if __name__ == "__main__":
    unittest.main()
