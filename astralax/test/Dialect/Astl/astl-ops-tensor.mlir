// RUN: astl-opt %s | astl-opt | FileCheck %s

// CHECK-LABEL: func.func @astl_dialect
func.func @astl_dialect(%arg0: tensor<5x4xf32>, %arg1: tensor<4x5xf32>, 
                       %arg2: tensor<5x5xf32>, %arg3: tensor<8x5x5xf32>) -> tensor<5x5xf32> {
  // CHECK: astl.identity
  %0 = astl.identity (%arg0: tensor<5x4xf32>) -> tensor<5x4xf32>
  // CHECK: astl.add
  %1 = astl.add (%0: tensor<5x4xf32>, %arg0: tensor<5x4xf32>) -> tensor<5x4xf32>
  // CHECK: astl.gemm
  %2 = astl.gemm (%arg0: tensor<5x4xf32>, %arg1: tensor<4x5xf32>, 
                   %arg2: tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: astl.brgemm
  %3 = astl.brgemm (%arg3: tensor<8x5x5xf32>, %arg3: tensor<8x5x5xf32>, 
                   %2: tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: astl.relu
  %4 = astl.relu (%3: tensor<5x5xf32>) -> tensor<5x5xf32>
  
  // CHECK: astl.zero
  %5 = astl.zero (%4: tensor<5x5xf32>) -> tensor<5x5xf32> 
  
  // CHECK: astl.zero {{.+}} {myAttr = "myattr"}
  %6 = astl.zero (%4: tensor<5x5xf32>) -> tensor<5x5xf32> {myAttr = "myattr"}
  return %5 : tensor<5x5xf32>
}

// CHECK-LABEL: func.func @astl_identity_tensor_bcast
func.func @astl_identity_tensor_bcast(%arg0: tensor<32xf32>) -> tensor<32x32xf32> {
  // CHECK: astl.identity
  %0 = astl.identity (%arg0: tensor<32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @astl_relu_tensor_scalar_bcast
func.func @astl_relu_tensor_scalar_bcast(%arg0: f32) -> tensor<32x32xf32> {
  // CHECK: astl.relu
  %0 = astl.relu (%arg0: f32) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @vnni_gemm_b_operand
func.func @vnni_gemm_b_operand(%arg0: tensor<32x32xbf16>, 
                               %arg1: tensor<16x32x2xbf16>,
                               %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // CHECK: astl.gemm
  %0 = astl.gemm (%arg0: tensor<32x32xbf16>, %arg1: tensor<16x32x2xbf16>,
                 %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0: tensor<32x32xbf16>
}

// CHECK-LABEL: func.func @fused_brgemm
func.func @fused_brgemm(%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, %arg2: tensor<32x32xf32>,
                        %bias: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: astl.fused_brgemm
  %0 = astl.fused_brgemm [unary = relu, binary = add] 
                        (%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, 
                         %arg2: tensor<32x32xf32>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func.func @fused_brgemm_with_1d_bias
func.func @fused_brgemm_with_1d_bias(%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, 
    %arg2: tensor<32x32xf32>, %bias: tensor<32xf32>) -> tensor<32x32xf32> {
  // CHECK: astl.fused_brgemm
  %0 = astl.fused_brgemm [unary = relu, binary = add]
                        (%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>,
                         %arg2: tensor<32x32xf32>, %bias: tensor<32xf32>) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}
