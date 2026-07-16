// RUN: ngopt-native-opt --ng-dce %s | FileCheck %s

module {
  func.func @dce(%x: tensor<2xf32>) -> tensor<2xf32> {
    %dead = ng.relu %x : tensor<2xf32>
    ng.return %x : tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @dce
// CHECK-NOT: ng.relu
// CHECK: ng.return %arg0
