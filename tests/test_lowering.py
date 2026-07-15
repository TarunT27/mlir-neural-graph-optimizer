from __future__ import annotations

import unittest

from ngopt.lowering import lower_to_linalg
from ngopt.parser import parse_module
from tests.test_parser import VALID_MODULE


class LoweringTests(unittest.TestCase):
    def test_supported_ops_lower_to_standard_dialect_text(self) -> None:
        module = parse_module(VALID_MODULE)

        lowered = lower_to_linalg(module.graph)

        self.assertIn("func.func @main", lowered)
        self.assertIn("arith.constant", lowered)
        self.assertIn("linalg.elemwise_binary", lowered)
        self.assertNotIn("ng.mul", lowered)


if __name__ == "__main__":
    unittest.main()
