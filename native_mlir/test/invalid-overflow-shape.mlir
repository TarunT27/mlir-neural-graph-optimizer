// RUN: not ngopt-native-opt %s 2>&1 | FileCheck %s

module {
  func.func @overflow(%x: tensor<4294967296x4294967296xf32>) -> tensor<1xf32> {
    %one = ng.constant dense<[1.0]> : tensor<1xf32>
    ng.return %one : tensor<1xf32>
  }
}

// CHECK: module exceeds the 16,000,000-element tensor safety budget
