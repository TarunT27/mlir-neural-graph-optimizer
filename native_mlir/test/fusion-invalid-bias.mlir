// RUN: not ngopt-native-opt --ng-fuse-linear-relu %s 2>&1 | FileCheck %s

module {
  func.func @broadcast_only(%x: tensor<1x2xf32>, %w: tensor<2x2xf32>, %b: tensor<1xf32>) -> tensor<1x2xf32> {
    %mm = ng.matmul %x, %w : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    %sum = ng.add %mm, %b : (tensor<1x2xf32>, tensor<1xf32>) -> tensor<1x2xf32>
    %out = ng.relu %sum : tensor<1x2xf32>
    ng.return %out : tensor<1x2xf32>
  }
}

// CHECK: operands are not broadcast compatible
