// RUN: not ngopt-native-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad(%x: tensor<1x3xf32>, %w: tensor<2x2xf32>) -> tensor<1x2xf32> {
    %out = ng.matmul %x, %w : (tensor<1x3xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    ng.return %out : tensor<1x2xf32>
  }
}

// CHECK: inner dimensions must match
