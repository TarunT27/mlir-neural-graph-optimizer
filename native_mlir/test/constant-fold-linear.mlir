// RUN: ngopt-native-opt --ng-constant-fold %s | FileCheck %s

module {
  func.func @linear() -> tensor<1x2xf32> {
    %input = ng.constant dense<[[3.0, -2.0]]> : tensor<1x2xf32>
    %weight = ng.constant dense<[[2.0, -1.0], [0.5, 3.0]]> : tensor<2x2xf32>
    %bias = ng.constant dense<[0.25, -0.5]> : tensor<2xf32>
    %out = ng.linear_relu %input, %weight, %bias : (tensor<1x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
    ng.return %out : tensor<1x2xf32>
  }
}

// CHECK: ng.constant dense<
// CHECK-SAME: 5.250000e+00
// CHECK-SAME: 0.000000e+00
// CHECK-NOT: ng.linear_relu
