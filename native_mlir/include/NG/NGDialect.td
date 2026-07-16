#ifndef NG_DIALECT
#define NG_DIALECT

include "mlir/IR/OpBase.td"

def NG_Dialect : Dialect {
  let name = "ng";
  let cppNamespace = "::mlir::ng";
  let summary = "A compact neural-graph optimization dialect";
  let description = [{
    The NG dialect models immutable, statically shaped f32 neural graphs and
    provides a focused path from graph rewrites to structured Linalg IR.
  }];
  let hasConstantMaterializer = 1;
}

class NG_Op<string mnemonic, list<Trait> traits = []>
    : Op<NG_Dialect, mnemonic, traits>;

#endif // NG_DIALECT
