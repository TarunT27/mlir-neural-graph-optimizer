from __future__ import annotations

import unittest
from pathlib import Path

from ngopt.ir import Module
from ngopt.native import print_native_module
from ngopt.parser import parse_module
from ngopt.passes import default_pipeline

ROOT = Path(__file__).resolve().parents[1]


class NativeBridgeTests(unittest.TestCase):
    def test_native_export_emits_functional_operand_types(self) -> None:
        module = parse_module((ROOT / "examples" / "tiny_mlp.mlir").read_text(encoding="utf-8"))

        text = print_native_module(module)

        self.assertIn(
            "ng.matmul %x, %weights : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>",
            text,
        )
        self.assertIn("dense<[[2.0, -1.0], [0.5, 3.0]]>", text)
        self.assertIn("ng.return %output : tensor<1x2xf32>", text)

    def test_native_export_supports_fused_operation(self) -> None:
        source = (ROOT / "examples" / "tiny_mlp.mlir").read_text(encoding="utf-8")
        module = parse_module(source)

        optimized = Module(module.name, default_pipeline().run(module.graph).graph)
        text = print_native_module(optimized)

        self.assertIn("ng.linear_relu", text)
        self.assertNotIn("ng.matmul", text)


if __name__ == "__main__":
    unittest.main()
