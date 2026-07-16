#include "NG/NGDialect.h"
#include "NG/NGOps.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::ng;

#include "NG/NGOpsDialect.cpp.inc"

void NGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NG/NGOps.cpp.inc"
      >();
}

Operation *NGDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location location) {
  auto elements = dyn_cast<ElementsAttr>(value);
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!elements || !tensorType || elements.getType() != tensorType)
    return nullptr;
  return ConstantOp::create(builder, location, tensorType, elements);
}
