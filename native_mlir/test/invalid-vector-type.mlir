// RUN: not ngopt-native-opt %s 2>&1 | FileCheck %s

module {
  func.func @invalid_vector(%x: vector<2xf32>, %y: vector<2xf32>) -> vector<2xf32> {
    %out = "ng.add"(%x, %y) : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    func.return %out : vector<2xf32>
  }
}

// CHECK: error:
