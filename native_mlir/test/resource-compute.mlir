// RUN: not ngopt-native-opt %s 2>&1 | FileCheck %s

module {
  func.func @too_much_work() -> tensor<1024x1024xf32> {
    %lhs = ng.constant dense<1.0> : tensor<1024x1024xf32>
    %rhs = ng.constant dense<1.0> : tensor<1024x1024xf32>
    %out = ng.matmul %lhs, %rhs : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    ng.return %out : tensor<1024x1024xf32>
  }
}

// CHECK: module exceeds the 1,000,000-operation compute safety limit
