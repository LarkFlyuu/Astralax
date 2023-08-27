// RUN: astl-opt %s -split-input-file -verify-diagnostics

func.func @astl_add_invalid(%arg0: f32, %arg1: f32) {
  // expected-error @below {{operand #2 must be 2D memref of floating-point values or 2D tensor of floating-point values, but got 'f32'}}
  astl.add ins(%arg0: f32, %arg0: f32) outs(%arg1: f32)
  return
}

// -----

func.func @astl_relu_invalid(%arg0: f32, %arg1: f32) {
  // expected-error @below {{operand #1 must be 2D memref of floating-point values or 2D tensor of floating-point values, but got 'f32'}}
  astl.relu ins(%arg0: f32) outs(%arg1: f32)
  return
}

// -----

func.func @astl_relu_invalid(%arg0: memref<f32>, %arg1: memref<f32>) {
  // expected-error @below {{operand #0 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values or floating-point, but got 'memref<f32>'}}
  astl.relu ins(%arg0: memref<f32>) outs(%arg1: memref<f32>)
  return
}

// -----

func.func @astl_zero_invalid(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{op fails to verify in-place computation}}
  astl.zero ins(%arg0: memref<2x2xf32>) outs(%arg1: memref<2x2xf32>)
  return %arg0 : memref<2x2xf32>
}

// -----

func.func @astl_add_invalid(%arg0: memref<f32>, %arg1: memref<f32>) {
  // expected-error @below {{operand #0 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values or floating-point, but got 'memref<f32>'}}
  astl.add ins(%arg0: memref<f32>, %arg1: memref<f32>) outs(%arg1: memref<f32>)
  return
}

// -----

func.func @astl_identity_invalid(%arg0: memref<1x2xf32>, %arg1: memref<2x2xf32>) -> memref<1x2xf32> {

  // expected-error @below {{op result type not broadcast compatible with broadcasted operands's shapes}}
  astl.identity ins(%arg1: memref<2x2xf32>) outs(%arg0: memref<1x2xf32>)
  return %arg0: memref<1x2xf32>
}

// -----

func.func @myfunc(%arg0: memref<?x?xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{operand #0 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values or floating-point, but got 'memref<?x?xf32>'}}
  astl.identity ins(%arg0: memref<?x?xf32>) outs(%arg1: memref<2x2xf32>)
  return %arg1: memref<2x2xf32>
}

// -----

func.func @astl_identity_invalid(%arg0: memref<3x3xf32>, %arg1: memref<2x3xf32>) -> memref<3x3xf32> {

  // expected-error @below {{result type not broadcast compatible with broadcasted operands's shapes}}
  astl.identity ins(%arg1: memref<2x3xf32>) outs(%arg0: memref<3x3xf32>)
  return %arg0: memref<3x3xf32>
}

// -----

func.func @astl_gemm_invalid(%arg0: memref<3x2xf32>, %arg1: memref<4x3xf32>,
                              %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // expected-error @below {{operand 0 fails to verify expected shape}}
  astl.gemm ins(%arg0: memref<3x2xf32>, %arg1: memref<4x3xf32>, %arg2: memref<5x5xf32>) 
           outs(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}

// -----

// The batch dimension must agree in both arg0 and arg1.
func.func @astl_brgemm_invalid(%arg0: memref<7x2x3xf32>, %arg1: memref<8x3x2xf32>,
                              %arg2: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{operand 1 fails to verify expected shape}}
  astl.brgemm ins(%arg0: memref<7x2x3xf32>, %arg1: memref<8x3x2xf32>, %arg2: memref<2x2xf32>) 
             outs(%arg2: memref<2x2xf32>)
  return %arg2: memref<2x2xf32>
}

// -----

func.func @astl_matmul_invalid(%arg0: memref<6x5xbf16>, %arg1: memref<5x6x2xbf16>,
                              %arg2: memref<6x6xbf16>) -> memref<6x6xbf16> {
  // expected-error @below {{operand 1 fails to verify expected shape}}
  astl.gemm ins(%arg0: memref<6x5xbf16>, %arg1: memref<5x6x2xbf16>, %arg2: memref<6x6xbf16>) 
           outs(%arg2: memref<6x6xbf16>)
  return %arg2: memref<6x6xbf16>
}

// -----

// Mixed types
func.func @astl_matmul_invalid(%arg0: memref<6x10xbf16>, %arg1: memref<5x6x2xbf16>,
                              %arg2: memref<6x6xf32>) -> memref<6x6xf32> {
  // expected-error @below {{requires the same element type for all operands}}
  astl.gemm ins(%arg0: memref<6x10xbf16>, %arg1: memref<5x6x2xbf16>, %arg2: memref<6x6xf32>) 
           outs(%arg2: memref<6x6xf32>)
  return %arg2: memref<6x6xf32>
}

// -----

func.func @astl_gemm_mixed_types(%arg0: memref<3x2xf32>, %arg1: memref<2x3xf32>,
                                %arg2: memref<3x3xbf16>) -> memref<3x3xbf16> {
  // expected-error @below {{requires the same element type for all operands}}
  astl.gemm ins(%arg0: memref<3x2xf32>, %arg1: memref<2x3xf32>, %arg2: memref<3x3xbf16>) 
             outs(%arg2: memref<3x3xbf16>)
  return %arg2: memref<3x3xbf16>
}

// -----

func.func @astl_add_check_broadcast_operand(%arg0: memref<2x3xf32>, %arg1: memref<3x3xf32>) {
  // expected-error @below {{operands don't have broadcast-compatible shapes}}
  astl.add ins(%arg0: memref<2x3xf32>, %arg1: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return
}

// -----

func.func @astl_add_check_broadcast_result(%arg0: memref<8x1xf32>, %arg1: memref<8x8xf32>) {
  // expected-error @below {{result type not broadcast compatible with broadcasted operands's shapes}}
  astl.add ins(%arg1: memref<8x8xf32>, %arg1: memref<8x8xf32>) outs(%arg0: memref<8x1xf32>)
  return 
}

// -----

func.func @astl_add_stride_inner_dim(%arg0: memref<8x8xf32, strided<[8, 2], offset: 0>>, 
                                    %arg1: memref<8x8xf32>) {
  // expected-error @below {{non-unit stride in the innermost varying dimension for operand 0}}
  astl.add ins(%arg0: memref<8x8xf32, strided<[8, 2], offset: 0>>, %arg1: memref<8x8xf32>) outs(%arg1: memref<8x8xf32>)
  return
}

// -----

func.func @astl_add_non_constant_stride(%arg0: memref<8x8xf32, strided<[?, ?], offset: 0>>,
                                       %arg1: memref<8x8xf32>) {
  // expected-error @below {{non-unit stride in the innermost varying dimension for operand 0}}
  astl.add ins(%arg0: memref<8x8xf32, strided<[?, ?], offset: 0>>, %arg1: memref<8x8xf32>) outs(%arg1: memref<8x8xf32>)
  return 
}

// -----

func.func @astl_add_mixing(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expect tensor type}}
  %0 = astl.add (%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  // expected-error @below {{expect tensor type}}
  %0 = astl.add (%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32>
  return %0 : memref<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expect tensor type}}
  %0 = astl.add (%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expected 'outs'}}
  %0 = astl.add ins(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expect memref type}}
  %0 = astl.add ins(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return %0 : tensor<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  // expected-error @below {{cannot name an operation with no results}}
  %0 = astl.add ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>) -> memref<3x3xf32>
  return %0 : memref<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) {
  // expected-error @below {{expected '->'}}
  astl.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) outs(%arg1: tensor<3x3xf32>)
  return %arg1 : tensor<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) {
  // expected-error @below {{expect memref type}}
  %0 = astl.add ins(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) outs(%arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> memref<3x3xf32> {
  // expected-error @below {{result #0 must be 2D tensor of floating-point values, but got 'memref<3x3xf32>'}}
  %0 = astl.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> memref<3x3xf32>
  return %0 : memref<3x3xf32>
}

// -----

func.func @astl_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) {
  // expected-error @below {{expect single result at tensor abstraction}}
  %0:2 = astl.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  return #0:2 : tensor<3x3xf32>
}

// -----

func.func @non_unit_stride_gemm(%arg0: memref<12x9xf32, strided<[?, ?], offset: ?>>,
    %arg1: memref<9x6xf32, strided<[?, ?], offset: ?>>,
    %arg2: memref<12x6xf32, strided<[?, ?], offset: ?>>) {
  // expected-error @below {{non-unit stride in the innermost varying dimension for operand 0}}
  astl.gemm ins(%arg0 : memref<12x9xf32, strided<[?, ?], offset: ?>>,
                 %arg1 : memref<9x6xf32, strided<[?, ?], offset: ?>>,
                 %arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
             outs(%arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
  return
}

// -----

func.func @astl_identity_invalid(%arg0: tensor<3x3xf32>) -> tensor<2x3xf32> {
  // expected-error @below {{op result type not broadcast compatible with broadcasted operands's shapes}}
  %0 = astl.identity (%arg0: tensor<3x3xf32>) -> tensor<2x3xf32>
  return %0: tensor<2x3xf32>
}

// -----

func.func @astl_add_check_broadcast_operand(%arg0: tensor<2x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{operands don't have broadcast-compatible shapes}}
  %0 = astl.add (%arg0: tensor<2x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @astl_gemm(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: tensor<2x2xf32>) {
  // expected-error @below {{expect memref type}}
  astl.gemm ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: tensor<2x2xf32>) 
             outs(%arg2: tensor<2x2xf32>)
  return
}

// -----

func.func @astl_add_invalid_number_of_operands(%arg0: memref<2x2xf32>) {
  // expected-error @below {{expect 2 input operands, but got: 3}}
  astl.add ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) outs(%arg0: memref<2x2xf32>)
  return
}

// -----

func.func @astl_relu_invalid_number_of_operands(%arg0: memref<2x2xf32>) {
  // expected-error @below {{expect 1 input operands, but got: 3}}
  astl.relu ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) outs(%arg0: memref<2x2xf32>)
  return
}

// -----

func.func @astl_gemm_invalid_number_of_operands(%arg0: memref<2x2xf32>) {
  // expected-error @below {{expect 3 input operands, but got: 2}}
  astl.gemm ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) 
             outs(%arg0: memref<2x2xf32>)
  return
}

// -----

func.func @astl_add_invalid_number_of_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 2 input operands, but got: 3}}
  %0 = astl.add (%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>
}

// -----

func.func @astl_relu_invalid_number_of_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 1 input operands, but got: 3}}
  %0 = astl.relu (%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>
}

// -----

func.func @astl_matmul_invalid_number_of_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 3 input operands, but got: 2}}
  %0 = astl.gemm (%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>
}

// -----

func.func @astl_segement_size(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 0 output operands, but got: 1}}
  %0 = astl.zero(%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {operand_segment_sizes = array<i32: 1, 1>}
  return %0 : tensor<2x2xf32>
}

// -----

func.func @unary_invalid_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 1 input operands, but got: 2}}
  %0 = astl.zero(%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @vnni_gemm_b_operand_wrong_type(%arg0: tensor<32x32xf32>,        
                                          %arg1: tensor<16x32x2xf32>,
                                          %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @below {{operand 1 invalid element type for VNNI layout expect bf16, but got: 'f32'}}
  %0 = astl.gemm (%arg0: tensor<32x32xf32>, %arg1: tensor<16x32x2xf32>,
                 %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}

// -----

func.func @vnni_wrong_layout(%arg0: tensor<32x32xbf16>,        
                             %arg1: tensor<16x32x3xbf16>,
                             %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // expected-error @below {{operand 1 invalid VNNI layout expect inner dims to be 2 or 4, but got: 3}}
  %0 = astl.gemm (%arg0: tensor<32x32xbf16>, %arg1: tensor<16x32x3xbf16>,
                 %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0: tensor<32x32xbf16>
}

// -----

func.func @vnni_gemm_b_operand_wrong_shape(%arg0: tensor<32x32xbf16>,        
                                           %arg1: tensor<15x32x2xbf16>,
                                           %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // expected-error @below {{operand 1 fails to verify expected shape}}
  %0 = astl.gemm (%arg0: tensor<32x32xbf16>, %arg1: tensor<15x32x2xbf16>,
                 %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0: tensor<32x32xbf16>
}

// -----

func.func @tied_type_result_to_arg2(%arg0: tensor<32x32xf32>,        
                                    %arg1: tensor<16x32x2xf32>,
                                    %arg2: tensor<1x32xf32>) -> tensor<32x32xf32> {
  // expected-error @below {{result type differs from destination operand type}}
  %0 = astl.gemm (%arg0: tensor<32x32xf32>, %arg1: tensor<16x32x2xf32>,
                 %arg2: tensor<1x32xf32>) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}

// -----

func.func @fused_brgemm_invalid_bias(%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, 
    %arg2: tensor<32x32xf32>, %bias: tensor<32x32x32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @below {{expected shaped type with rank 1 or 2 for bias}}
  %0 = astl.fused_brgemm [unary = relu, binary = add]
                        (%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>,
                         %arg2: tensor<32x32xf32>, 
                         %bias: tensor<32x32x32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// TODO: (lorenzo) we should handle scalar type too but requires a bit of work
// during xsmm conversion, and an overloading function in the runtime.
func.func @fused_brgemm_scalar_bias(%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>,
    %arg2: tensor<32x32xf32>, %bias: f32) -> tensor<32x32xf32> {
  // expected-error @below {{must be 1D/2D/3D/4D memref of floating-point values or 1D/2D/3D/4D tensor of floating-point values, but got 'f32'}}
  %0 = astl.fused_brgemm [unary = relu, binary = add]
                        (%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>,
                         %arg2: tensor<32x32xf32>, %bias: f32) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}
