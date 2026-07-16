// RUN: ngopt-native-opt --ng-canonicalize %s | FileCheck %s

module {
  func.func @canonicalize(%x: tensor<2xf32>) -> tensor<2xf32> {
    %one = ng.constant dense<1.0> : tensor<2xf32>
    %mul = ng.mul %x, %one : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %relu0 = ng.relu %mul : tensor<2xf32>
    %relu1 = ng.relu %relu0 : tensor<2xf32>
    ng.return %relu1 : tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @canonicalize
// CHECK-NOT: ng.mul
// CHECK: %[[R:.*]] = ng.relu %arg0
// CHECK-NOT: ng.relu %[[R]]
// CHECK: ng.return %[[R]]
