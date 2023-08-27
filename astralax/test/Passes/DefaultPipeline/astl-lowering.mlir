// RUN: astl-opt %s -astl-lowering | FileCheck %s -check-prefix=XSMM
// RUN: astl-opt %s -astl-lowering="astl-to-loops" | FileCheck %s -check-prefix=LOOPS

func.func @astl_ops(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {
    astl.brgemm ins(%arg0 : memref<3x5x4xf32>, %arg1 : memref<3x4x5xf32>, %arg2 : memref<5x5xf32>) 
               outs(%arg2 : memref<5x5xf32>)
    astl.relu ins(%arg2 : memref<5x5xf32>) outs(%arg2 : memref<5x5xf32>)
    astl.gemm ins(%arg2 : memref<5x5xf32>, %arg3 : memref<5x5xf32>, %arg2 : memref<5x5xf32>) 
             outs(%arg2 : memref<5x5xf32>)
    return
  }

// XSMM-LABEL: func.func @astl_ops(
// XSMM-NOT: astl.brgemm
// XSMM: xsmm.brgemm
// XSMM-NOT: astl.relu
// XSMM: xsmm.unary relu
// XSMM-NOT: astl.gemm
// XSMM: xsmm.gemm

// LOOPS-LABEL: func.func @astl_ops(
// LOOPS-NOT: astl.brgemm
// LOOPS: scf.for
// LOOPS:   arith.mulf
// LOOPS:   arith.addf
// LOOPS-NOT: astl.relu
// LOOPS: scf.for
// LOOPS:   arith.maxf
// LOOPS-NOT: astl.gemm
// LOOPS: scf.for
// LOOPS:   arith.mulf
// LOOPS:   arith.addf

// XSMM-LABEL: copy
func.func @copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  // XSMM: xsmm.unary.dispatch identity
  // XSMM-NEXT: xsmm.unary identity
  memref.copy %arg0, %arg1 : memref<2x2xf32> to memref<2x2xf32>
  return
}
