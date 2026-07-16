// RUN: ngopt-native-opt --ng-constant-fold %s | FileCheck %s

module {
  func.func @keep_finite_boundary() -> tensor<1xf32> {
    %max = ng.constant dense<[3.40282347E+38]> : tensor<1xf32>
    %two = ng.constant dense<[2.0]> : tensor<1xf32>
    %out = ng.mul %max, %two : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    ng.return %out : tensor<1xf32>
  }
}

// CHECK-LABEL: func.func @keep_finite_boundary
// CHECK: ng.mul
// CHECK-NOT: dense<[0x7F800000]>
