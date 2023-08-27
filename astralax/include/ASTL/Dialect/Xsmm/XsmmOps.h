//===- XsmmOps.h - Xsmm dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_DIALECT_XSMM_XSMMOPS_H
#define ASTL_DIALECT_XSMM_XSMMOPS_H

#include "ASTL/Dialect/Xsmm/XsmmDialect.h"
#include "ASTL/Dialect/Xsmm/XsmmEnum.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "ASTL/Dialect/Xsmm/XsmmOps.h.inc"

#endif // ASTL_DIALECT_XSMM_XSMMOPS_H
