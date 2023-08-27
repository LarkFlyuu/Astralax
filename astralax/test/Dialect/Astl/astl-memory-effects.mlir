// RUN: astl-opt %s -cse | FileCheck %s

// CHECK-LABEL: pure_at_tensor
func.func @pure_at_tensor(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) {
  // CHECK-NOT: astl.add
  %0 = astl.add(%arg0 : tensor<2x2xf32>, %arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NOT: astl.zero
  %1 = astl.zero (%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NOT: astl.gemm
  %2 = astl.gemm (%arg0 : tensor<2x2xf32>, %arg1 : tensor<2x2xf32>, %arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
