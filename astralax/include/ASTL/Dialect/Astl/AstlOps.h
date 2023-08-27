//===- AstlOps.h - astl dialect ops -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_DIALECT_ASTL_ASTLOPS_H
#define ASTL_DIALECT_ASTL_ASTLOPS_H

#include "ASTL/Dialect/Astl/AstlAttr.h"
#include "ASTL/Dialect/Astl/AstlDialect.h"
#include "ASTL/Dialect/Astl/AstlInterface.h"
#include "ASTL/Dialect/Astl/AstlTraits.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "ASTL/Dialect/Astl/AstlOps.h.inc"

#endif // ASTL_DIALECT_ASTL_ASTLOPS_H
