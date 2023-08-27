//===- AstlDialect.td - astl dialect ------------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_ASTL_DIALECT
#define ASTL_ASTL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ASTL/Dialect/Astl/AstlInterface.td"

//===----------------------------------------------------------------------===//
// ASTL dialect definition.
//===----------------------------------------------------------------------===//

def Astl_Dialect : Dialect {
    let name = "astl";
    let summary = "Tensor Processing Primitives dialect.";
    let description = [{
        This dialect reflects the Tensor Processing Primitives
        paper: https://arxiv.org/abs/2104.05755.
    }];
    let cppNamespace = "::mlir::astl";
    let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// astl Traits
//===----------------------------------------------------------------------===//

// astl operation trait - make sure the operand are broadcastable.
// Two dimensions are compatible when:
// 1. they are equal, or
// 2. one of them is 1.
def BroadcastableShape : NativeOpTrait<"astl::BroadcastableShape">;

// astl operation trait - make sure the stride in the fastest-varying
// dimension is one.
def UnitStrideInnerLoop : NativeOpTrait<"astl::UnitStrideInnerLoop">;

// astl operation trait - verify astl operand segment size.
def UnaryOp : NativeOpTrait<"astl::UnaryOp">;
def BinaryOp : NativeOpTrait<"astl::BinaryOp">;
def TernaryOp : NativeOpTrait<"astl::TernaryOp">;
def QuaternaryOp : NativeOpTrait<"astl::QuaternaryOp">;

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Astl_Op<string mnemonic, list<Trait> traits = []> :
        Op<Astl_Dialect, mnemonic, !listconcat(traits, [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                UnitStrideInnerLoop,   
                SameOperandsElementType,
                AstlOpInterface])>;

#endif // ASTL_ASTL_DIALECT
