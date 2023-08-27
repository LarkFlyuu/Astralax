// RUN: astl-opt %s -gpu-conversion -split-input-file | FileCheck %s

func.func @astl_identity(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {
  astl.identity ins(%arg0: memref<5x1xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @astl_identity
// CHECK:         gpu.launch_func  @astl_identity_kernel::@astl_identity_kernel
// CHECK: gpu.module @astl_identity_kernel
// CHECK-LABEL: gpu.func @astl_identity_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.load
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @astl_relu(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  astl.relu ins(%arg0: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @astl_relu
// CHECK:         gpu.launch_func  @astl_relu_kernel::@astl_relu_kernel
// CHECK: gpu.module @astl_relu_kernel
// CHECK-LABEL: gpu.func @astl_relu_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.load
// CHECK:         arith.maxf
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @astl_zero(%arg0: memref<3x3xf32>) {
  astl.zero ins(%arg0: memref<3x3xf32>) outs(%arg0: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @astl_zero
// CHECK:         gpu.launch_func  @astl_zero_kernel::@astl_zero_kernel
// CHECK: gpu.module @astl_zero_kernel
// CHECK-LABEL: gpu.func @astl_zero_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @astl_add(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  astl.add ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @astl_add
// CHECK:         gpu.launch_func  @astl_add_kernel::@astl_add_kernel
// CHECK: gpu.module @astl_add_kernel
// CHECK-LABEL: gpu.func @astl_add_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.load
// CHECK:         memref.load
// CHECK:         arith.addf
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @astl_brgemm(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) {
  astl.brgemm ins(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>)
             outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @astl_brgemm
// CHECK:         gpu.launch_func  @astl_brgemm_kernel::@astl_brgemm_kernel
// CHECK: gpu.module @astl_brgemm_kernel
// CHECK-LABEL: gpu.func @astl_brgemm_kernel
// CHECK:         gpu.block_id
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             memref.load
// CHECK:             memref.load
// CHECK:             memref.load
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:             memref.store
// CHECK:         gpu.return

// -----

func.func @astl_gemm(%arg0: memref<8x9xf32>, %arg1: memref<9x10xf32>, %arg2: memref<8x10xf32>) {
  astl.gemm ins(%arg0 : memref<8x9xf32>, %arg1 : memref<9x10xf32>, %arg2: memref<8x10xf32>)
           outs(%arg2: memref<8x10xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @astl_gemm
// CHECK:         gpu.launch_func  @astl_gemm_kernel::@astl_gemm_kernel
// CHECK: gpu.module @astl_gemm_kernel
// CHECK-LABEL: gpu.func @astl_gemm_kernel
// CHECK:         gpu.block_id
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           memref.store
// CHECK:         gpu.return
