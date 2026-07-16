cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0219)
  cmake_policy(SET CMP0219 NEW)
endif()
project(ngopt-native LANGUAGES C CXX)

find_package(MLIR 22.1.6 EXACT REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

enable_testing()
add_test(
  NAME ngopt-native-tests
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/test/run_native_tests.py
          --driver $<TARGET_FILE:ngopt-native-opt>
          --filecheck ${LLVM_TOOLS_BINARY_DIR}/FileCheck${CMAKE_EXECUTABLE_SUFFIX}
          --test-dir ${PROJECT_SOURCE_DIR}/test
)
add_test(
  NAME ngopt-reproducer-disabled
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/test/assert_reproducer_disabled.py
          --driver $<TARGET_FILE:ngopt-native-opt>
          --input ${PROJECT_SOURCE_DIR}/test/dialect.mlir
)
add_custom_target(check-ng
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS ngopt-native-opt
  USES_TERMINAL
)
