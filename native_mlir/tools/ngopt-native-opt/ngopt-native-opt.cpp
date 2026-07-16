#include "NG/NGDialect.h"
#include "NG/NGPasses.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace {
constexpr uint64_t kMaximumInputBytes = 4 * 1024 * 1024;

std::unique_ptr<llvm::MemoryBuffer>
readInput(llvm::StringRef inputFilename) {
  if (inputFilename == "-") {
    llvm::errs() << "ngopt-native-opt: stdin is disabled; provide a regular "
                    "file no larger than 4 MiB\n";
    return nullptr;
  }
  auto opened = llvm::sys::fs::openNativeFileForRead(inputFilename);
  if (!opened) {
    llvm::errs() << "ngopt-native-opt: cannot open input: "
                 << llvm::toString(opened.takeError()) << "\n";
    return nullptr;
  }
  llvm::sys::fs::file_t descriptor = *opened;
  llvm::sys::fs::file_status status;
  std::error_code statusError = llvm::sys::fs::status(descriptor, status);
  if (statusError) {
    llvm::errs() << "ngopt-native-opt: cannot inspect input: "
                 << statusError.message() << "\n";
    llvm::sys::fs::closeFile(descriptor);
    return nullptr;
  }
  if (!llvm::sys::fs::is_regular_file(status)) {
    llvm::errs() << "ngopt-native-opt: input must be a regular file\n";
    llvm::sys::fs::closeFile(descriptor);
    return nullptr;
  }
  const uint64_t size = status.getSize();
  if (size > kMaximumInputBytes) {
    llvm::errs() << "ngopt-native-opt: input exceeds the 4 MiB safety limit\n";
    llvm::sys::fs::closeFile(descriptor);
    return nullptr;
  }
  auto buffer = llvm::MemoryBuffer::getOpenFile(
      descriptor, inputFilename, size, /*RequiresNullTerminator=*/true,
      /*IsVolatile=*/true);
  std::error_code closeError = llvm::sys::fs::closeFile(descriptor);
  if (!buffer) {
    llvm::errs() << "ngopt-native-opt: cannot read input: "
                 << buffer.getError().message() << "\n";
    return nullptr;
  }
  if (closeError) {
    llvm::errs() << "ngopt-native-opt: cannot close input: "
                 << closeError.message() << "\n";
    return nullptr;
  }
  return std::move(*buffer);
}
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  mlir::ng::registerNGPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::ng::NGDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect>();

  std::string inputFilename;
  std::string outputFilename;
  std::tie(inputFilename, outputFilename) = mlir::registerAndParseCLIOptions(
      argc, argv, "MLIR neural-graph optimizer\n", registry);
  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();
  if (config.shouldRunReproducer()) {
    llvm::errs() << "ngopt-native-opt: --run-reproducer is disabled because "
                    "it can bypass mandatory resource guards\n";
    return EXIT_FAILURE;
  }
  config.verifyOnParsing(true).allowUnregisteredDialects(false).verifyPasses(
      true);
  mlir::MlirOptMainConfig userConfig = config;
  config.setPassPipelineSetupFn(
      [&userConfig](mlir::PassManager &manager) -> mlir::LogicalResult {
        manager.addPass(mlir::ng::createResourceGuardPass());
        return userConfig.setupPassPipeline(manager);
      });

  auto input = readInput(inputFilename);
  if (!input) {
    return EXIT_FAILURE;
  }
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }
  if (failed(mlir::MlirOptMain(output->os(), std::move(input), registry,
                               config)))
    return EXIT_FAILURE;
  output->keep();
  return EXIT_SUCCESS;
}
