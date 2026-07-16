// RUN: not ngopt-native-opt %s 2>&1 | FileCheck %s

module {
  func.func @too_large() -> tensor<1048576xf32> {
    %c00 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c01 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c02 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c03 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c04 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c05 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c06 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c07 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c08 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c09 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c10 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c11 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c12 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c13 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c14 = ng.constant dense<0.0> : tensor<1048576xf32>
    %c15 = ng.constant dense<0.0> : tensor<1048576xf32>
    ng.return %c15 : tensor<1048576xf32>
  }
}

// CHECK: module exceeds the 16,000,000-element tensor safety budget
