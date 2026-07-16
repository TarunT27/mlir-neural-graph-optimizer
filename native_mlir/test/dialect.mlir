// RUN: ngopt-native-opt %s | FileCheck %s

module {
  func.func @graph(%x: tensor<1x2xf32>, %w: tensor<2x2xf32>, %b: tensor<2xf32>) -> tensor<1x2xf32> {
    %mm = ng.matmul %x, %w : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    %sum = ng.add %mm, %b : (tensor<1x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
    %out = ng.relu %sum : tensor<1x2xf32>
    ng.return %out : tensor<1x2xf32>
  }
}

// CHECK-LABEL: func.func @graph
// CHECK: ng.matmul
// CHECK: ng.add
// CHECK: ng.relu
// CHECK: ng.return
