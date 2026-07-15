from __future__ import annotations

import unittest

from ngopt.parser import ParseError, parse_module
from ngopt.printer import print_module

VALID_MODULE = """module @tiny {
  func.func @main(%x: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %one = ng.constant dense<[1.0, 1.0]> : tensor<1x2xf32>
    %out = ng.mul %x, %one : tensor<1x2xf32>
    ng.return %out : tensor<1x2xf32>
  }
}
"""


class ParserTests(unittest.TestCase):
    def test_parse_print_round_trip_is_stable(self) -> None:
        first = parse_module(VALID_MODULE, source="tiny.mlir")
        rendered = print_module(first)
        second = parse_module(rendered, source="roundtrip.mlir")

        self.assertEqual(first, second)
        self.assertEqual(rendered, print_module(second))

    def test_malformed_input_reports_source_and_line(self) -> None:
        with self.assertRaisesRegex(ParseError, r"bad\.mlir:2:"):
            parse_module("module @bad {\n  this is not mlir\n}", source="bad.mlir")

    def test_unknown_operation_is_rejected(self) -> None:
        invalid = VALID_MODULE.replace("ng.mul", "ng.teleport")

        with self.assertRaisesRegex(ParseError, "unsupported operation"):
            parse_module(invalid, source="unknown.mlir")

    def test_function_and_return_type_mismatch_is_rejected(self) -> None:
        invalid = VALID_MODULE.replace(
            "func.func @main(%x: tensor<1x2xf32>) -> tensor<1x2xf32>",
            "func.func @main(%x: tensor<1x2xf32>) -> tensor<2xf32>",
        )

        with self.assertRaisesRegex(ParseError, "function result type"):
            parse_module(invalid, source="mismatch.mlir")

    def test_operation_after_return_is_rejected(self) -> None:
        invalid = VALID_MODULE.replace(
            "    ng.return %out : tensor<1x2xf32>",
            "    ng.return %out : tensor<1x2xf32>\n    %late = ng.relu %out : tensor<1x2xf32>",
        )

        with self.assertRaisesRegex(ParseError, "after ng.return"):
            parse_module(invalid, source="late.mlir")


if __name__ == "__main__":
    unittest.main()
