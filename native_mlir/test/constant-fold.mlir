// RUN: ngopt-native-opt --ng-constant-fold %s | FileCheck %s

module {
  func.func @fold() -> tensor<2xf32> {
    %a = ng.constant dense<[1.0, 2.0]> : tensor<2xf32>
    %b = ng.constant dense<[3.0, 4.0]> : tensor<2xf32>
    %sum = ng.add %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %out = ng.relu %sum : tensor<2xf32>
    ng.return %out : tensor<2xf32>
  }
}

// CHECK: ng.constant dense<[4.000000e+00, 6.000000e+00]>
// CHECK-NOT: ng.add
// CHECK-NOT: ng.relu
