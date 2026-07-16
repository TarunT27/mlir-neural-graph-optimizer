set(LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIROptLib
  MLIRTensorDialect
  NGIR
)

add_llvm_executable(ngopt-native-opt ngopt-native-opt.cpp)
llvm_update_compile_flags(ngopt-native-opt)
target_link_libraries(ngopt-native-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(ngopt-native-opt)
