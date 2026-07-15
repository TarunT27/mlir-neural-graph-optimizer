module @tiny_mlp {
  func.func @main(%x: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %weights = ng.constant dense<[2.0, -1.0, 0.5, 3.0]> : tensor<2x2xf32>
    %bias = ng.constant dense<[0.25, -0.5]> : tensor<2xf32>
    %matmul = ng.matmul %x, %weights : tensor<1x2xf32>
    %biased = ng.add %matmul, %bias : tensor<1x2xf32>
    %output = ng.relu %biased : tensor<1x2xf32>
    ng.return %output : tensor<1x2xf32>
  }
}
