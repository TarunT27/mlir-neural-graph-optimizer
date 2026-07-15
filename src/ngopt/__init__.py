"""A compact, dependency-free neural graph optimizer with MLIR-style SSA text."""

from ngopt.ir import Graph, Module, Operation, Tensor, TensorType, Value
from ngopt.passes import PassManager, default_pipeline
from ngopt.runtime import execute

__all__ = [
    "Graph",
    "Module",
    "Operation",
    "PassManager",
    "Tensor",
    "TensorType",
    "Value",
    "default_pipeline",
    "execute",
]

__version__ = "0.1.0"
