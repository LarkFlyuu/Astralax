//===- XsmmDialect.cpp - Xsmm dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Dialect/Xsmm/XsmmDialect.h"
#include "ASTL/Dialect/Xsmm/XsmmOps.h"

using namespace mlir;
using namespace mlir::xsmm;

//===----------------------------------------------------------------------===//
// Xsmm dialect.
//===----------------------------------------------------------------------===//

void XsmmDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ASTL/Dialect/Xsmm/XsmmOps.cpp.inc"
      >();
}

#include "ASTL/Dialect/Xsmm/XsmmOpsDialect.cpp.inc"
