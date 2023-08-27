// RUN: astl-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// Check if astl ops on tensors can be successfully bufferized by
// the default pipeline and then executed.
// The main goal for this test is to validate that custom bufferization
// interfaces are registered and active in astl-run.
func.func @entry(%arg0: tensor<4x4xf32>, %out: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant dense<0.01> : tensor<4x4xf32>
  %0 = tensor.empty() : tensor<4x4xf32>
  %1 = astl.zero (%0 : tensor<4x4xf32>) -> (tensor<4x4xf32>)
  // destination style used due to memory leaks
  // TODO: stop using %out tensor when astl-run kernel return value
  //       deallocation is working correctly
  %2 = astl.gemm (%arg0 : tensor<4x4xf32>, %cst : tensor<4x4xf32>, %out : tensor<4x4xf32>) -> (tensor<4x4xf32>)
  return %2 : tensor<4x4xf32>
}

// CHECK: ( 1.04, 1.04, 1.04, 1.04 )
// CHECK: ( 1.04, 1.04, 1.04, 1.04 )
// CHECK: ( 1.04, 1.04, 1.04, 1.04 )
// CHECK: ( 1.04, 1.04, 1.04, 1.04 )
