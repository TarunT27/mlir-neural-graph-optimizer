from __future__ import annotations

import unittest

from ngopt.ir import Graph, Operation, Tensor, TensorType, Value
from ngopt.verify import VerificationError, verify_graph


class TensorTests(unittest.TestCase):
    def test_tensor_rejects_wrong_element_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 4 elements"):
            Tensor(TensorType((2, 2)), (1.0, 2.0))

    def test_tensor_type_rejects_dynamic_or_zero_dimensions(self) -> None:
        for shape in ((0, 2), (-1, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TensorType(shape)

    def test_ir_normalizes_caller_owned_collections_to_tuples(self) -> None:
        source_shape = [2]
        source_operands = ["%x"]
        source_outputs = ["%out"]
        tensor_type = TensorType(source_shape)  # type: ignore[arg-type]
        operation = Operation(
            "%out",
            "ng.relu",
            source_operands,
            tensor_type,  # type: ignore[arg-type]
        )
        graph = Graph(
            "main",
            [Value("%x", tensor_type)],  # type: ignore[arg-type]
            [operation],  # type: ignore[arg-type]
            source_outputs,  # type: ignore[arg-type]
        )

        source_shape[0] = 99
        source_operands[0] = "%changed"
        source_outputs[0] = "%changed"

        self.assertEqual(tensor_type.shape, (2,))
        self.assertEqual(operation.operands, ("%x",))
        self.assertEqual(graph.outputs, ("%out",))

    def test_non_finite_tensor_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            Tensor(TensorType((1,)), (float("inf"),))


class VerifierTests(unittest.TestCase):
    def test_valid_graph_verifies(self) -> None:
        tensor_type = TensorType((1, 2))
        graph = Graph(
            "main",
            (Value("%x", tensor_type),),
            (
                Operation.make_constant("%one", Tensor(tensor_type, (1.0, 1.0))),
                Operation("%out", "ng.mul", ("%x", "%one"), tensor_type),
            ),
            ("%out",),
        )

        self.assertIs(verify_graph(graph), graph)

    def test_use_before_definition_is_rejected(self) -> None:
        tensor_type = TensorType((1,))
        graph = Graph(
            "main",
            (),
            (Operation("%out", "ng.relu", ("%missing",), tensor_type),),
            ("%out",),
        )

        with self.assertRaisesRegex(VerificationError, "%missing.*not defined"):
            verify_graph(graph)

    def test_matmul_shape_mismatch_is_rejected(self) -> None:
        lhs_type = TensorType((2, 3))
        rhs_type = TensorType((4, 2))
        output_type = TensorType((2, 2))
        graph = Graph(
            "main",
            (Value("%lhs", lhs_type), Value("%rhs", rhs_type)),
            (Operation("%out", "ng.matmul", ("%lhs", "%rhs"), output_type),),
            ("%out",),
        )

        with self.assertRaisesRegex(VerificationError, "inner dimensions"):
            verify_graph(graph)

    def test_duplicate_ssa_result_is_rejected(self) -> None:
        tensor_type = TensorType((1,))
        graph = Graph(
            "main",
            (Value("%x", tensor_type),),
            (Operation("%x", "ng.relu", ("%x",), tensor_type),),
            ("%x",),
        )

        with self.assertRaisesRegex(VerificationError, "duplicate SSA"):
            verify_graph(graph)

    def test_unsupported_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            TensorType((1,), "i8")

    def test_oversized_tensor_is_rejected_before_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety limit"):
            TensorType((2_000, 2_000))

    def test_excessive_matmul_work_is_rejected(self) -> None:
        matrix = TensorType((400, 400))
        graph = Graph(
            "main",
            (Value("%lhs", matrix), Value("%rhs", matrix)),
            (Operation("%out", "ng.matmul", ("%lhs", "%rhs"), matrix),),
            ("%out",),
        )

        with self.assertRaisesRegex(VerificationError, "compute safety limit"):
            verify_graph(graph)

    def test_input_only_graph_still_obeys_cumulative_memory_limit(self) -> None:
        large = TensorType((600_000,))
        graph = Graph(
            "main",
            (Value("%a", large), Value("%b", large)),
            (),
            ("%a",),
        )

        with self.assertRaisesRegex(VerificationError, "memory safety limit"):
            verify_graph(graph)

    def test_multiple_outputs_are_rejected(self) -> None:
        tensor_type = TensorType((1,))
        graph = Graph(
            "main",
            (Value("%a", tensor_type), Value("%b", tensor_type)),
            (),
            ("%a", "%b"),
        )

        with self.assertRaisesRegex(VerificationError, "exactly one"):
            verify_graph(graph)


if __name__ == "__main__":
    unittest.main()
