// RUN: astl-opt %s -convert-memref-to-astl -split-input-file | FileCheck %s

func.func @copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  memref.copy %arg0, %arg1 : memref<2x2xf32> to memref<2x2xf32>
  return
}

// CHECK-LABEL: copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32>, %[[ARG1:.+]]: memref<2x2xf32>
// CHECK: astl.identity ins(%[[ARG0]] : memref<2x2xf32>) 
// CHECK-SAME: outs(%[[ARG1]] : memref<2x2xf32>)

// -----

func.func @strided_copy(%arg0: memref<2x2xf32, strided<[2, 1], offset: ?>>,
                        %arg1: memref<2x2xf32, strided<[2, 1], offset: ?>>) {
  memref.copy %arg0, %arg1 : memref<2x2xf32, strided<[2, 1], offset: ?>> to memref<2x2xf32, strided<[2, 1], offset: ?>>
  return
}

// CHECK-LABEL: strided_copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32, strided<[2, 1], offset: ?>>, %[[ARG1:.+]]: memref<2x2xf32, strided<[2, 1], offset: ?>>
// CHECK: astl.identity ins(%[[ARG0]] : memref<2x2xf32, strided<[2, 1], offset: ?>>)
// CHECK-SAME: outs(%[[ARG1]] : memref<2x2xf32, strided<[2, 1], offset: ?>>)

// -----

// CHECK-LABEL: copy_non_unit_stride
func.func @copy_non_unit_stride(%arg0: memref<4x2xf32, strided<[2, 2], offset: ?>>,
                        %arg1: memref<4x2xf32, strided<[2, 1], offset: ?>>) {
  // CHECK-NOT: astl.identity
  // CHECK: memref.copy
  memref.copy %arg0, %arg1 : memref<4x2xf32, strided<[2, 2], offset: ?>> to memref<4x2xf32, strided<[2, 1], offset: ?>>
  return
}

// -----

// CHECK-LABEL: copy_3d
func.func @copy_3d(%arg0: memref<2x2x2xf32>, %arg1: memref<2x2x2xf32>) {
  // CHECK-NOT: astl.identity
  memref.copy %arg0, %arg1 : memref<2x2x2xf32> to memref<2x2x2xf32>
  return
}
