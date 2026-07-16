#ifndef NG_OPS
#define NG_OPS

include "NG/NGDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NG_ConstantOp : NG_Op<"constant", [Pure, ConstantLike,
                                        AllTypesMatch<["value", "result"]>]> {
  let summary = "Create a statically shaped f32 tensor constant";
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def NG_AddOp : NG_Op<"add", [Pure]> {
  let summary = "Broadcasting tensor addition";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def NG_MulOp : NG_Op<"mul", [Pure]> {
  let summary = "Broadcasting tensor multiplication";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def NG_MatmulOp : NG_Op<"matmul", [Pure]> {
  let summary = "Rank-two matrix multiplication";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def NG_ReluOp : NG_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise rectified linear unit";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def NG_LinearReluOp : NG_Op<"linear_relu", [Pure]> {
  let summary = "Fused matrix multiplication, bias addition, and ReLU";
  let arguments = (ins AnyTensor:$input, AnyTensor:$weight, AnyTensor:$bias);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input `,` $weight `,` $bias attr-dict `:` `(` type($input) `,` type($weight) `,` type($bias) `)` `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def NG_ReturnOp : NG_Op<"return", [Pure, ReturnLike, Terminator]> {
  let summary = "Return one tensor from a graph function";
  let arguments = (ins AnyTensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasVerifier = 1;
}

#endif // NG_OPS
