//===- AstlOps.td - astl dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_ASTL_OPS
#define ASTL_ASTL_OPS

include "ASTL/Dialect/Astl/AstlDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "ASTL/Dialect/Astl/AstlAttr.td"

class StaticMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

class StaticTensorRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[TensorOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
      TensorOf<allowedTypes>.summary,
      "::mlir::RankedTensorType">;

def AstlMemRefInput : StaticMemRefRankOf<[AnyFloat], [1, 2]>;
def AstlTensorInput : StaticTensorRankOf<[AnyFloat], [1, 2]>;
def AstlMemRefOutput : StaticMemRefRankOf<[AnyFloat], [2]>;
def AstlTensorOutput : StaticTensorRankOf<[AnyFloat], [2]>;

def AstlGemmLikeMemRef : StaticMemRefRankOf<[AnyFloat], [1, 2, 3, 4]>;
def AstlGemmLikeTensor : StaticTensorRankOf<[AnyFloat], [1, 2, 3, 4]>;

// astl operands:
// input operand: is a scalar float or a static memref with rank 1 or 2.
// output operand: static memref with rank 1 or 2.
def AstlInputOperand : AnyTypeOf<[AstlMemRefInput, AstlTensorInput, AnyFloat]>;
def AstlOutputOperand : AnyTypeOf<[AstlMemRefOutput, AstlTensorOutput]>;

// astl operands for gemm and brgemm ops.
def AstlGemmLikeOperand : AnyTypeOf<[AstlGemmLikeMemRef, AstlGemmLikeTensor]>;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

class Astl_UnaryOp<string mnemonic, list<Trait> traits = []> :
  Astl_Op<mnemonic, !listconcat(traits, [BroadcastableShape,
                                        UnaryOp])> {
  
  let arguments = (ins Variadic<AstlInputOperand>:$inputs, 
                       Variadic<AstlOutputOperand>:$outputs);
  let results = (outs Variadic<AstlTensorOutput>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1; 

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output)>,
    OpBuilder<(ins "Value":$input, "Type":$output)>
  ]; 
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

def Astl_IdentityOp : Astl_UnaryOp<"identity"> {
  let summary = "Copies input to output.";
  let description = [{
    The `astl.identity` copies input memref to output memref. It supports
    Numpy-style broadcast. 
   
    Example:

    ```mlir

    // out-of-place - memref abstraction.
    astl.identity ins(%1: memref<2x2xf32>) outs(%2: memref<2x2xf32>)
    
    // bcast - memref abstraction.
    astl.identity ins(%1: f32) outs(%2: memref<2x2xf32>)

    // tensor abstraction.
    %0 = astl.identity (%1: tensor<3x3xf32>) -> tensor<3x3xf32>

    ```
  }];
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def Astl_ReluOp : Astl_UnaryOp<"relu"> {
  let summary = "Applies a Rectified Linear Unit function in place.";
  let description = [{
    The `astl.relu` applies a Rectified Linear Unit function in place 
    or out-of-place. It supports Numpy-style broadcast.

    Example:

    ```mlir

    // out-of-place - memref abstraction.
    astl.relu ins(%0: memref<2x2xf32>) outs(%1: memref<2x2xf32>)

    // in-place - memref abstraction.
    astl.relu ins(%0: memref<2x2xf32>) outs(%0: memref<2x2xf32>)

    // bcast - memref abstraction.
    astl.relu ins(%0: memref<4xf32>) outs(%1: memref<2x4xf32>)

    // tensor abstraction.
    %0 = astl.relu (%0: tensor<4xf32>) -> tensor<4xf32>

    ```
  }];
}

//===----------------------------------------------------------------------===//
// ZeroOp
//===----------------------------------------------------------------------===//

def Astl_ZeroOp : Astl_UnaryOp<"zero"> {
  let summary = "Zero a tensor or memref.";
  let description = [{
    Zero initialize a tensor or memref value.
    
    Example:
    
    ```mlir
    
    // in-place - memref abstraction.
    astl.zero ins(%0: memref<2x2xf32>) outs(%0: memref<2x2xf32>)

    // tensor abstraction.
    %0 = astl.zero (%0: tensor<4xf32>) -> tensor<4xf32>
    
    ```
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

class Astl_BinaryOp<string mnemonic, list<Trait> traits = []> :
  Astl_Op<mnemonic, !listconcat(traits, [BinaryOp,
                                        BroadcastableShape])> {

  let arguments = (ins Variadic<AstlInputOperand>:$inputs, 
                       Variadic<AstlOutputOperand>:$outputs);
  let results = (outs Variadic<AstlTensorOutput>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>,
    OpBuilder<(ins "ValueRange":$inputs, "Type":$output)>
  ];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Astl_AddOp : Astl_BinaryOp<"add"> {
  let summary = "Element-wise addition.";
  let description = [{
    The `astl.add` operation performs element-wise addition on two-dimensional
    memrefs or ranked tensors, writing the result on the output memref (or in the
    result at tensor abstraction). At memref, no checks or assumption are made on
    the input/output arguments so the same memref can be passed both as input and
    output. At tensor level the operation produces a new tensor. In both cases, the
    op supports broadcast semantic see `BroadcastableShape` rules. 

    Example:

    ```mlir

    // A = A + A - memref abstraction.
    astl.add ins(%1: memref<2x2xf32>, %1: memref<2x2xf32>)
            outs(%1: memref<2x2xf32>)

    // B = A + B - memref abstraction.
    astl.add ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>) 
            outs(%2: memref<2x2xf32>)

    // C = A + B - memref abstraction.
    astl.add ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>)
            outs(%3: memref<2x2xf32>)

    // bcast.
    astl.add ins(%1: memref<1x3xf32>, %2: memref<3xf32>)
            outs(%3: memref<3x3xf32>) 

    // tensor abstraction.
    astl.add (%1: tensor<3x3xf32>, %2: tensor<3x3xf32>) -> tensor<3x3xf32>

    ```
  }];
}

//===----------------------------------------------------------------------===//
// Ternary Operations
//===----------------------------------------------------------------------===//

class Astl_TernaryOp<string mnemonic, list<Trait> traits = []> :
  Astl_Op<mnemonic, !listconcat(traits, [TernaryOp])> {

  let arguments = (ins Variadic<AstlGemmLikeOperand>:$inputs,
                       Variadic<AstlGemmLikeOperand>:$outputs); 
  let results = (outs Variadic<AstlGemmLikeOperand>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>,
    OpBuilder<(ins "ValueRange":$inputs, "Type":$output_type)>
  ]; 
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//

def Astl_GemmOp : Astl_TernaryOp<"gemm"> {
  let summary = "Performs matrix multiplication of two input.";
  let description = [{
    The `astl.gemm` mirrors `linalg.matmul`.
    C = beta * C + A * B (beta = 1).

    Example:

    ```mlir

    // Memref abstraction.
    astl.gemm ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>)
             outs(%3: memref<2x2xf32>)

    // Tensor abstraction.
    %0 = astl.gemm(%1: tensor<2x2xf32>, %2: tensor<2x2xf32>, 
                  %3: tensor<2x2xf32>) -> tensor<2x2xf32>

    // Tensor abstraction with VNNI layout on B.
    %0 = astl.gemm(%1: tensor<32x32xbf16>, %2: tensor<16x32x2xbf16>
                  %3: tensor<32x32xf32>) -> tensor<32x32xf32>

    ```
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

def Astl_BrgemmOp : Astl_TernaryOp<"brgemm"> {
  let summary = "Performs batch reduced matrix multiplication of two inputs.";
  let description = [{
    The `astl.brgemm` is an implementation of the Batch GEMM operation in oneAPI.

    Example:

    ```mlir

      // Memref abstraction.
      astl.brgemm ins(%1: memref<3x5x4xf32>, %2: memref<3x4x5xf32>)
                 outs(%3: memref<5x5xf32>)

      // Tensor abstraction.
      %0 = astl.brgemm (%1: tensor<3x5x4xf32>, %2: tensor<3x4x5xf32> 
                       %3: tensor<5x5xf32>) -> tensor<5x5xf32>
    ```
  }];
 
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quaternary Operations
//===----------------------------------------------------------------------===//

class Astl_QuaternaryOp<string mnemonic, list<Trait> traits = []> :
  Astl_Op<mnemonic, !listconcat(traits, [QuaternaryOp])> {

  let arguments = (ins Variadic<AstlGemmLikeOperand>:$inputs,
                       Variadic<AstlGemmLikeOperand>:$outputs); 
  let results = (outs Variadic<AstlGemmLikeOperand>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1; 
}

//===----------------------------------------------------------------------===//
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

def Astl_FusedBrgemmOp : Astl_QuaternaryOp<"fused_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication. On the result performs
    a binary operation specified by `binary_type` followed by a unary
    operation specified in `unary_type`.
  }];

  let description = [{
    The `astl.fused_brgemm` is an implementation of the Batch GEMM operation 
    in oneAPI.

    Example:

    ```mlir

      astl.fused_brgemm [unary = relu, binary = add] 
                       ins(%1: memref<3x6x4xf32>, %2: memref<3x4x4xf32>, 
                           %3: memref<4xf32>, %4: i64)
                       outs(%5: memref<6x4xf32>)
    ```
  }];

  let arguments = (ins Variadic<AstlGemmLikeOperand>:$inputs,
                       Variadic<AstlGemmLikeOperand>:$outputs, 
                       Astl_FusedUnaryOpKind:$unary_kind,
                       Astl_FusedBinaryOpKind:$binary_kind);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output, "Value":$bias, 
                   "FusedUnaryOpKindAttr":$unary_kind,
                   "FusedBinaryOpKindAttr":$binary_kind)>,
    OpBuilder<(ins "ValueRange":$inputs, "Type":$output_type, "Value":$bias,
                   "FusedUnaryOpKindAttr":$unary_kind,
                   "FusedBinaryOpKindAttr":$binary_kind)>
  ];
  let extraClassDeclaration = [{
    // Get the operand to be used as input to the binary operation.
    Value getBiasOperand() { return getInputs()[3]; };
  }];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

#endif // ASTL_ASTL_OPS
