//===- AstlAttr.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_DIALECT_ASTL_ASTLATTR_H
#define ASTL_DIALECT_ASTL_ASTLATTR_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "ASTL/Dialect/Astl/AstlAttr.h.inc"

#endif // ASTL_DIALECT_ASTL_ASTLATTR_H
