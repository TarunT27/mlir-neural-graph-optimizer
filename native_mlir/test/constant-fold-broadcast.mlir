// RUN: ngopt-native-opt --ng-constant-fold %s | FileCheck %s

module {
  func.func @broadcast() -> tensor<2x2xf32> {
    %matrix = ng.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %bias = ng.constant dense<[10.0, 20.0]> : tensor<2xf32>
    %out = ng.add %matrix, %bias : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    ng.return %out : tensor<2x2xf32>
  }
}

// CHECK: ng.constant dense<
// CHECK-SAME: 1.100000e+01
// CHECK-SAME: 2.200000e+01
// CHECK-SAME: 1.300000e+01
// CHECK-SAME: 2.400000e+01
// CHECK-NOT: ng.add
