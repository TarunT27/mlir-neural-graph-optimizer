#include "NG/NGPasses.h"
#include "NG/NGOps.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::ng;

namespace {
enum class ElementwiseKind { Add, Multiply, Relu };

AffineMap createBroadcastMap(OpBuilder &builder, RankedTensorType inputType,
                             RankedTensorType resultType) {
  SmallVector<AffineExpr> expressions;
  const int64_t rankOffset = resultType.getRank() - inputType.getRank();
  for (int64_t index = 0; index < inputType.getRank(); ++index) {
    const int64_t outputIndex = rankOffset + index;
    if (inputType.getDimSize(index) == 1 &&
        resultType.getDimSize(outputIndex) != 1)
      expressions.push_back(builder.getAffineConstantExpr(0));
    else
      expressions.push_back(builder.getAffineDimExpr(outputIndex));
  }
  return AffineMap::get(resultType.getRank(), 0, expressions,
                        builder.getContext());
}

Value createElementwise(PatternRewriter &rewriter, Location location,
                        ValueRange inputs, RankedTensorType resultType,
                        ElementwiseKind kind) {
  Value empty = tensor::EmptyOp::create(
      rewriter, location, resultType.getShape(), resultType.getElementType());

  SmallVector<AffineMap> maps;
  maps.reserve(inputs.size() + 1);
  for (Value input : inputs)
    maps.push_back(createBroadcastMap(
        rewriter, cast<RankedTensorType>(input.getType()), resultType));
  maps.push_back(rewriter.getMultiDimIdentityMap(resultType.getRank()));

  SmallVector<utils::IteratorType> iterators(
      resultType.getRank(), utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, location, resultType, inputs, ValueRange{empty}, maps,
      iterators,
      [kind](OpBuilder &builder, Location bodyLocation, ValueRange arguments) {
        Value result;
        switch (kind) {
        case ElementwiseKind::Add:
          result = arith::AddFOp::create(builder, bodyLocation, arguments[0],
                                        arguments[1]);
          break;
        case ElementwiseKind::Multiply:
          result = arith::MulFOp::create(builder, bodyLocation, arguments[0],
                                        arguments[1]);
          break;
        case ElementwiseKind::Relu: {
          Value zero = arith::ConstantFloatOp::create(
              builder, bodyLocation, builder.getF32Type(), APFloat(0.0F));
          result = arith::MaxNumFOp::create(builder, bodyLocation, arguments[0],
                                           zero);
          break;
        }
        }
        linalg::YieldOp::create(builder, bodyLocation, result);
      });
  return generic.getResult(0);
}

Value createMatmul(PatternRewriter &rewriter, Location location, Value lhs,
                   Value rhs, RankedTensorType resultType) {
  Value zero = arith::ConstantFloatOp::create(
      rewriter, location, rewriter.getF32Type(), APFloat(0.0F));
  Value empty = tensor::EmptyOp::create(
      rewriter, location, resultType.getShape(), resultType.getElementType());
  Value initialized =
      linalg::FillOp::create(rewriter, location, zero, empty).getResult(0);
  return linalg::MatmulOp::create(rewriter, location, resultType,
                                 ValueRange{lhs, rhs}, ValueRange{initialized})
      .getResult(0);
}

struct LowerConstant final : OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, op.getResult().getType(), op.getValue());
    return success();
  }
};

struct LowerAdd final : OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOp(op, createElementwise(rewriter, op.getLoc(),
                                             {op.getLhs(), op.getRhs()},
                                             resultType,
                                             ElementwiseKind::Add));
    return success();
  }
};

struct LowerMul final : OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOp(op, createElementwise(rewriter, op.getLoc(),
                                             {op.getLhs(), op.getRhs()},
                                             resultType,
                                             ElementwiseKind::Multiply));
    return success();
  }
};

struct LowerRelu final : OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOp(op,
                       createElementwise(rewriter, op.getLoc(), op.getInput(),
                                         resultType, ElementwiseKind::Relu));
    return success();
  }
};

struct LowerMatmul final : OpRewritePattern<MatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOp(op, createMatmul(rewriter, op.getLoc(), op.getLhs(),
                                       op.getRhs(), resultType));
    return success();
  }
};

struct LowerLinearRelu final : OpRewritePattern<LinearReluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LinearReluOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value matmul = createMatmul(rewriter, op.getLoc(), op.getInput(),
                                op.getWeight(), resultType);
    Value biased = createElementwise(rewriter, op.getLoc(),
                                     {matmul, op.getBias()}, resultType,
                                     ElementwiseKind::Add);
    rewriter.replaceOp(
        op, createElementwise(rewriter, op.getLoc(), biased, resultType,
                              ElementwiseKind::Relu));
    return success();
  }
};

struct LowerReturn final : OpRewritePattern<ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getInput());
    return success();
  }
};

struct ConvertNGToLinalgPass final
    : PassWrapper<ConvertNGToLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNGToLinalgPass)

  StringRef getArgument() const final { return "convert-ng-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower the NG dialect completely to Arith, Tensor, Linalg, and Func";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerConstant, LowerAdd, LowerMul, LowerMatmul, LowerRelu,
                 LowerLinearRelu, LowerReturn>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    bool illegalOperation = false;
    getOperation().walk([&](Operation *operation) {
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace() == "ng") {
        operation->emitError("NG operation survived full lowering");
        illegalOperation = true;
      }
    });
    if (illegalOperation)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::ng::createConvertNGToLinalgPass() {
  return std::make_unique<ConvertNGToLinalgPass>();
}
