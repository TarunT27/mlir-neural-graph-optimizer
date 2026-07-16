add_mlir_dialect_library(NGIR
  NGDialect.cpp
  NGOps.cpp
  NGPasses.cpp
  NGLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NG

  DEPENDS
  MLIRNGOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransforms
)
