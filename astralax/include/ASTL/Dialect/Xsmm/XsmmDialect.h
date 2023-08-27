//===- XsmmDialect.h - Xsmm dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSMM_ASTL_DIALECT_H
#define XSMM_ASTL_DIALECT_H

#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "ASTL/Dialect/Xsmm/XsmmOpsDialect.h.inc"

#endif // XSMM_ASTL_DIALECT_H
