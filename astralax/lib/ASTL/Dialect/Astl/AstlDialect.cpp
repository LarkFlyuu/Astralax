//===- AstlDialect.cpp - astl dialect ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Dialect/Astl/AstlDialect.h"
#include "ASTL/Dialect/Astl/AstlOps.h"

using namespace mlir;
using namespace mlir::astl;

//===----------------------------------------------------------------------===//
// astl dialect.
//===----------------------------------------------------------------------===//

void AstlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ASTL/Dialect/Astl/AstlOps.cpp.inc"
      >();
}

#include "ASTL/Dialect/Astl/AstlOpsDialect.cpp.inc"
