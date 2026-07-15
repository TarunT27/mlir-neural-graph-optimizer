module @constant_fold {
  func.func @main() -> tensor<2xf32> {
    %lhs = ng.constant dense<[1.0, -3.0]> : tensor<2xf32>
    %rhs = ng.constant dense<[2.0, 4.0]> : tensor<2xf32>
    %sum = ng.add %lhs, %rhs : tensor<2xf32>
    %output = ng.relu %sum : tensor<2xf32>
    ng.return %output : tensor<2xf32>
  }
}
