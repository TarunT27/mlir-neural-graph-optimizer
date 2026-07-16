#include "NG/NGOps.h"
#include "NG/NGDialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::ng;

namespace {
constexpr int64_t kMaximumTensorElements = 1'048'576;

FailureOr<RankedTensorType> requireTensor(Operation *op, Type type,
                                          StringRef role) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return op->emitOpError() << role << " must be a ranked tensor";
  if (!tensor.hasStaticShape())
    return op->emitOpError() << role << " must have a static shape";
  if (!tensor.getElementType().isF32())
    return op->emitOpError() << role << " must use f32 elements";
  if (tensor.getRank() > 2)
    return op->emitOpError() << role << " rank must be between 0 and 2";
  int64_t elementCount = 1;
  for (int64_t dimension : tensor.getShape()) {
    if (dimension <= 0)
      return op->emitOpError() << role << " dimensions must be positive";
    if (dimension > kMaximumTensorElements ||
        elementCount > kMaximumTensorElements / dimension)
      return op->emitOpError() << role << " exceeds the element limit of "
                               << kMaximumTensorElements;
    elementCount *= dimension;
  }
  return tensor;
}

FailureOr<SmallVector<int64_t>> broadcastShape(RankedTensorType lhs,
                                               RankedTensorType rhs) {
  if (lhs == rhs)
    return SmallVector<int64_t>(lhs.getShape());
  if (lhs.getRank() == 2 && rhs.getRank() == 1 &&
      lhs.getDimSize(1) == rhs.getDimSize(0))
    return SmallVector<int64_t>(lhs.getShape());
  if (rhs.getRank() == 2 && lhs.getRank() == 1 &&
      rhs.getDimSize(1) == lhs.getDimSize(0))
    return SmallVector<int64_t>(rhs.getShape());
  return failure();
}

LogicalResult verifyBroadcastBinary(Operation *op, Type lhsType, Type rhsType,
                                    Type resultType) {
  auto lhs = requireTensor(op, lhsType, "left operand");
  auto rhs = requireTensor(op, rhsType, "right operand");
  auto result = requireTensor(op, resultType, "result");
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  auto shape = broadcastShape(*lhs, *rhs);
  if (failed(shape))
    return op->emitOpError("operands are not broadcast compatible");
  if (ArrayRef<int64_t>(*shape) != result->getShape())
    return op->emitOpError("result shape does not match broadcast shape");
  return success();
}

int64_t broadcastOffset(RankedTensorType inputType,
                        RankedTensorType resultType, int64_t resultOffset) {
  SmallVector<int64_t> coordinates(resultType.getRank(), 0);
  int64_t remaining = resultOffset;
  for (int64_t index = resultType.getRank() - 1; index >= 0; --index) {
    coordinates[index] = remaining % resultType.getDimSize(index);
    remaining /= resultType.getDimSize(index);
  }
  const int64_t rankOffset = resultType.getRank() - inputType.getRank();
  int64_t inputOffset = 0;
  for (int64_t index = 0; index < inputType.getRank(); ++index) {
    const int64_t dimension = inputType.getDimSize(index);
    const int64_t coordinate =
        dimension == 1 ? 0 : coordinates[rankOffset + index];
    inputOffset = inputOffset * dimension + coordinate;
  }
  return inputOffset;
}

DenseElementsAttr foldBinary(DenseElementsAttr lhs, DenseElementsAttr rhs,
                             RankedTensorType resultType, bool multiply) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType)
    return {};
  SmallVector<APFloat> lhsValues = llvm::to_vector(lhs.getValues<APFloat>());
  SmallVector<APFloat> rhsValues = llvm::to_vector(rhs.getValues<APFloat>());
  SmallVector<APFloat> values;
  values.reserve(resultType.getNumElements());
  for (int64_t offset = 0; offset < resultType.getNumElements(); ++offset) {
    APFloat value = lhsValues[broadcastOffset(lhsType, resultType, offset)];
    const APFloat &rhsValue =
        rhsValues[broadcastOffset(rhsType, resultType, offset)];
    if (multiply)
      value.multiply(rhsValue, APFloat::rmNearestTiesToEven);
    else
      value.add(rhsValue, APFloat::rmNearestTiesToEven);
    if (!value.isFinite())
      return {};
    values.push_back(value);
  }
  return DenseElementsAttr::get(resultType, values);
}

DenseElementsAttr foldMatmul(DenseElementsAttr lhs, DenseElementsAttr rhs,
                             RankedTensorType resultType) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2)
    return {};
  const int64_t rows = lhsType.getDimSize(0);
  const int64_t inner = lhsType.getDimSize(1);
  const int64_t columns = rhsType.getDimSize(1);
  SmallVector<APFloat> lhsValues = llvm::to_vector(lhs.getValues<APFloat>());
  SmallVector<APFloat> rhsValues = llvm::to_vector(rhs.getValues<APFloat>());
  SmallVector<APFloat> values;
  values.reserve(resultType.getNumElements());
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < columns; ++column) {
      APFloat sum = APFloat::getZero(APFloat::IEEEsingle());
      for (int64_t index = 0; index < inner; ++index) {
        APFloat product = lhsValues[row * inner + index];
        product.multiply(rhsValues[index * columns + column],
                         APFloat::rmNearestTiesToEven);
        sum.add(product, APFloat::rmNearestTiesToEven);
      }
      if (!sum.isFinite())
        return {};
      values.push_back(sum);
    }
  }
  return DenseElementsAttr::get(resultType, values);
}

DenseElementsAttr foldRelu(DenseFPElementsAttr input,
                           RankedTensorType resultType) {
  SmallVector<APFloat> values;
  values.reserve(resultType.getNumElements());
  for (APFloat value : input.getValues<APFloat>()) {
    if (value.isNegative())
      value = APFloat::getZero(value.getSemantics());
    values.push_back(value);
  }
  return DenseElementsAttr::get(resultType, values);
}

bool isAllOnes(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return false;
  auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  if (!dense)
    return false;
  return llvm::all_of(dense.getValues<APFloat>(),
                      [](const APFloat &item) { return item.isExactlyValue(1.0); });
}

struct FoldMulOne final : OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs().getType() == op.getResult().getType() &&
        isAllOnes(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (op.getRhs().getType() == op.getResult().getType() &&
        isAllOnes(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

struct FoldNestedRelu final : OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getInput().getDefiningOp<ReluOp>();
    if (!inner || inner.getResult().getType() != op.getResult().getType())
      return failure();
    rewriter.replaceOp(op, inner.getResult());
    return success();
  }
};
} // namespace

LogicalResult ConstantOp::verify() {
  auto result = requireTensor(*this, getResult().getType(), "result");
  if (failed(result))
    return failure();
  if (getValue().getType() != *result)
    return emitOpError("value type must match result type");
  auto values = dyn_cast<DenseFPElementsAttr>(getValue());
  if (!values)
    return emitOpError("value must contain only finite f32 numbers");
  if (values.isSplat()) {
    if (!values.getSplatValue<APFloat>().isFinite())
      return emitOpError("value must contain only finite f32 numbers");
  } else if (llvm::any_of(values.getValues<APFloat>(),
                          [](const APFloat &value) {
                            return !value.isFinite();
                          })) {
    return emitOpError("value must contain only finite f32 numbers");
  }
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

LogicalResult AddOp::verify() {
  return verifyBroadcastBinary(*this, getLhs().getType(), getRhs().getType(),
                               getResult().getType());
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhs || !rhs || !result)
    return {};
  return foldBinary(lhs, rhs, result, false);
}

LogicalResult MulOp::verify() {
  return verifyBroadcastBinary(*this, getLhs().getType(), getRhs().getType(),
                               getResult().getType());
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhs || !rhs || !result)
    return {};
  return foldBinary(lhs, rhs, result, true);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<FoldMulOne>(context);
}

LogicalResult MatmulOp::verify() {
  auto lhs = requireTensor(*this, getLhs().getType(), "left operand");
  auto rhs = requireTensor(*this, getRhs().getType(), "right operand");
  auto result = requireTensor(*this, getResult().getType(), "result");
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  if (lhs->getRank() != 2 || rhs->getRank() != 2 || result->getRank() != 2)
    return emitOpError("matmul requires rank-two operands and result");
  if (lhs->getDimSize(1) != rhs->getDimSize(0))
    return emitOpError("inner dimensions must match");
  if (result->getShape() !=
      ArrayRef<int64_t>({lhs->getDimSize(0), rhs->getDimSize(1)}))
    return emitOpError("result shape must be [lhs rows, rhs columns]");
  return success();
}

OpFoldResult MatmulOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhs || !rhs || !result)
    return {};
  return foldMatmul(lhs, rhs, result);
}

LogicalResult ReluOp::verify() {
  if (failed(requireTensor(*this, getInput().getType(), "input")) ||
      failed(requireTensor(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  return success();
}

OpFoldResult ReluOp::fold(FoldAdaptor adaptor) {
  auto input = dyn_cast_or_null<DenseFPElementsAttr>(adaptor.getInput());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!input || !result)
    return {};
  return foldRelu(input, result);
}

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<FoldNestedRelu>(context);
}

LogicalResult LinearReluOp::verify() {
  auto input = requireTensor(*this, getInput().getType(), "input");
  auto weight = requireTensor(*this, getWeight().getType(), "weight");
  auto bias = requireTensor(*this, getBias().getType(), "bias");
  auto result = requireTensor(*this, getResult().getType(), "result");
  if (failed(input) || failed(weight) || failed(bias) || failed(result))
    return failure();
  if (input->getRank() != 2 || weight->getRank() != 2 ||
      bias->getRank() != 1 || result->getRank() != 2)
    return emitOpError("linear_relu expects rank-2 input/weight/result and rank-1 bias");
  if (input->getDimSize(1) != weight->getDimSize(0))
    return emitOpError("input and weight inner dimensions must match");
  const int64_t columns = weight->getDimSize(1);
  if (bias->getDimSize(0) != columns)
    return emitOpError("bias length must match output columns");
  if (result->getDimSize(0) != input->getDimSize(0) ||
      result->getDimSize(1) != columns)
    return emitOpError("result shape does not match linear output");
  return success();
}

OpFoldResult LinearReluOp::fold(FoldAdaptor adaptor) {
  auto input = dyn_cast_or_null<DenseElementsAttr>(adaptor.getInput());
  auto weight = dyn_cast_or_null<DenseElementsAttr>(adaptor.getWeight());
  auto bias = dyn_cast_or_null<DenseElementsAttr>(adaptor.getBias());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!input || !weight || !bias || !resultType)
    return {};
  DenseElementsAttr matmul = foldMatmul(input, weight, resultType);
  if (!matmul)
    return {};
  DenseElementsAttr biased = foldBinary(matmul, bias, resultType, false);
  auto biasedFP = dyn_cast_or_null<DenseFPElementsAttr>(biased);
  if (!biasedFP)
    return {};
  return foldRelu(biasedFP, resultType);
}

LogicalResult ReturnOp::verify() {
  if (failed(requireTensor(*this, getInput().getType(), "input")))
    return failure();
  auto function = (*this)->getParentOfType<func::FuncOp>();
  if (!function)
    return emitOpError("must be nested in func.func");
  if (function.getNumResults() != 1)
    return emitOpError("requires a function with exactly one result");
  if (function.getResultTypes().front() != getInput().getType())
    return emitOpError("operand type must match the function result");
  return success();
}

#define GET_OP_CLASSES
#include "NG/NGOps.cpp.inc"
