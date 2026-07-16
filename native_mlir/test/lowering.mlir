// RUN: ngopt-native-opt --convert-ng-to-linalg %s | FileCheck %s

module {
  func.func @lower(%x: tensor<1x2xf32>, %w: tensor<2x2xf32>, %b: tensor<2xf32>) -> tensor<1x2xf32> {
    %out = ng.linear_relu %x, %w, %b : (tensor<1x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
    ng.return %out : tensor<1x2xf32>
  }
}

// CHECK-LABEL: func.func @lower
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.maxnumf
// CHECK: return
// CHECK-NOT: ng.
