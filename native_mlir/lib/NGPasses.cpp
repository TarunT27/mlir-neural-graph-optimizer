#include "NG/NGPasses.h"
#include "NG/NGOps.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::ng;

namespace {
constexpr int64_t kMaximumOperations = 10'000;
constexpr int64_t kMaximumCumulativeElements = 16'000'000;
constexpr int64_t kMaximumComputeWork = 1'000'000;

FailureOr<int64_t> checkedElements(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return failure();
  int64_t count = 1;
  for (int64_t dimension : tensor.getShape()) {
    if (dimension <= 0 || dimension > kMaximumCumulativeElements ||
        count > kMaximumCumulativeElements / dimension)
      return failure();
    count *= dimension;
  }
  return count;
}

struct ResourceGuardPass final
    : PassWrapper<ResourceGuardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResourceGuardPass)

  StringRef getArgument() const final { return "ng-resource-guard"; }
  StringRef getDescription() const final {
    return "Enforce native module operation and tensor budgets";
  }
  void runOnOperation() final {
    int64_t operationCount = 0;
    int64_t cumulativeElements = 0;
    int64_t cumulativeWork = 0;
    bool invalid = false;
    getOperation().walk([&](Operation *operation) {
      if (invalid)
        return;
      if (++operationCount > kMaximumOperations) {
        operation->emitError("module exceeds the 10,000-operation safety limit");
        invalid = true;
        return;
      }
      auto account = [&](Type type) {
        if (!isa<TensorType>(type))
          return;
        auto elements = checkedElements(type);
        if (failed(elements) ||
            *elements > kMaximumCumulativeElements - cumulativeElements) {
          operation->emitError(
              "module exceeds the 16,000,000-element tensor safety budget");
          invalid = true;
          return;
        }
        cumulativeElements += *elements;
      };
      for (Type type : operation->getResultTypes())
        account(type);
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            account(argument.getType());
      if (invalid)
        return;
      auto consumeWork = [&](ArrayRef<int64_t> factors) {
        int64_t work = 1;
        for (int64_t factor : factors) {
          if (factor <= 0 || factor > kMaximumComputeWork ||
              work > kMaximumComputeWork / factor) {
            operation->emitError(
                "module exceeds the 1,000,000-operation compute safety limit");
            invalid = true;
            return;
          }
          work *= factor;
        }
        if (work > kMaximumComputeWork - cumulativeWork) {
          operation->emitError(
              "module exceeds the 1,000,000-operation compute safety limit");
          invalid = true;
          return;
        }
        cumulativeWork += work;
      };
      if (auto matmul = dyn_cast<MatmulOp>(operation)) {
        auto lhs = dyn_cast<RankedTensorType>(matmul.getLhs().getType());
        auto rhs = dyn_cast<RankedTensorType>(matmul.getRhs().getType());
        if (!lhs || !rhs || lhs.getRank() != 2 || rhs.getRank() != 2) {
          operation->emitError(
              "resource guard requires ranked tensor matmul operands");
          invalid = true;
          return;
        }
        consumeWork(
            {lhs.getDimSize(0), lhs.getDimSize(1), rhs.getDimSize(1)});
      } else if (auto linear = dyn_cast<LinearReluOp>(operation)) {
        auto input = dyn_cast<RankedTensorType>(linear.getInput().getType());
        auto weight = dyn_cast<RankedTensorType>(linear.getWeight().getType());
        if (!input || !weight || input.getRank() != 2 ||
            weight.getRank() != 2) {
          operation->emitError(
              "resource guard requires ranked tensor linear_relu operands");
          invalid = true;
          return;
        }
        consumeWork({input.getDimSize(0), input.getDimSize(1),
                     weight.getDimSize(1)});
      } else if (isa<AddOp, MulOp, ReluOp>(operation)) {
        auto result =
            dyn_cast<RankedTensorType>(operation->getResult(0).getType());
        if (!result) {
          operation->emitError(
              "resource guard requires ranked tensor elementwise results");
          invalid = true;
          return;
        }
        SmallVector<int64_t> factors(result.getShape());
        consumeWork(factors);
      }
    });
    if (invalid)
      signalPassFailure();
  }
};

struct ConstantFoldPass final
    : PassWrapper<ConstantFoldPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldPass)

  StringRef getArgument() const final { return "ng-constant-fold"; }
  StringRef getDescription() const final {
    return "Fold constant NG operations with IEEE binary32 semantics";
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct CanonicalizePass final
    : PassWrapper<CanonicalizePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizePass)

  StringRef getArgument() const final { return "ng-canonicalize"; }
  StringRef getDescription() const final {
    return "Apply type-safe NG graph canonicalizations";
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    MulOp::getCanonicalizationPatterns(patterns, &getContext());
    ReluOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct FuseLinearReluPattern final : OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp relu,
                                PatternRewriter &rewriter) const final {
    auto add = relu.getInput().getDefiningOp<AddOp>();
    if (!add || !add->hasOneUse())
      return failure();

    MatmulOp matmul;
    Value bias;
    if ((matmul = add.getLhs().getDefiningOp<MatmulOp>()))
      bias = add.getRhs();
    else if ((matmul = add.getRhs().getDefiningOp<MatmulOp>()))
      bias = add.getLhs();
    else
      return failure();

    if (!matmul->hasOneUse())
      return failure();
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    auto resultType = dyn_cast<RankedTensorType>(relu.getResult().getType());
    if (!biasType || biasType.getRank() != 1 || !resultType ||
        resultType.getRank() != 2 ||
        biasType.getDimSize(0) != resultType.getDimSize(1))
      return failure();

    rewriter.replaceOpWithNewOp<LinearReluOp>(
        relu, relu.getResult().getType(), matmul.getLhs(), matmul.getRhs(),
        bias);
    return success();
  }
};

struct FuseLinearReluPass final
    : PassWrapper<FuseLinearReluPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseLinearReluPass)

  StringRef getArgument() const final { return "ng-fuse-linear-relu"; }
  StringRef getDescription() const final {
    return "Fuse one-use matmul, bias add, and ReLU graphs";
  }
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseLinearReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct DCEPass final : PassWrapper<DCEPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DCEPass)

  StringRef getArgument() const final { return "ng-dce"; }
  StringRef getDescription() const final {
    return "Erase unused side-effect-free NG operations";
  }
  void runOnOperation() final {
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *operation) {
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace() == "ng")
        candidates.push_back(operation);
    });
    for (Operation *operation : llvm::reverse(candidates)) {
      if (!isa<ReturnOp>(operation) && isOpTriviallyDead(operation))
        operation->erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::ng::createConstantFoldPass() {
  return std::make_unique<ConstantFoldPass>();
}

std::unique_ptr<Pass> mlir::ng::createCanonicalizePass() {
  return std::make_unique<CanonicalizePass>();
}

std::unique_ptr<Pass> mlir::ng::createFuseLinearReluPass() {
  return std::make_unique<FuseLinearReluPass>();
}

std::unique_ptr<Pass> mlir::ng::createDCEPass() {
  return std::make_unique<DCEPass>();
}

std::unique_ptr<Pass> mlir::ng::createResourceGuardPass() {
  return std::make_unique<ResourceGuardPass>();
}

void mlir::ng::registerNGPasses() {
  registerPass([] { return createConstantFoldPass(); });
  registerPass([] { return createCanonicalizePass(); });
  registerPass([] { return createFuseLinearReluPass(); });
  registerPass([] { return createDCEPass(); });
  registerPass([] { return createConvertNGToLinalgPass(); });
}
