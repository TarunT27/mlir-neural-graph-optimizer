// RUN: not ngopt-native-opt %s 2>&1 | FileCheck %s

module {
  func.func @nonfinite() -> tensor<1xf32> {
    %nan = ng.constant dense<[0x7FC00000]> : tensor<1xf32>
    ng.return %nan : tensor<1xf32>
  }
}

// CHECK: value must contain only finite f32 numbers
