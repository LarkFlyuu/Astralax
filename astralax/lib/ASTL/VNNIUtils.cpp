//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/VNNIUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace vnni {
namespace utils {

std::optional<int64_t> getVnniBlockingFactor(Type type) {
  auto elementType = getElementTypeOrSelf(type);
  return std::nullopt;
}

bool isInVnniLayout(MemRefType memref) {
  if (memref.getRank() < 3 || !memref.getElementType().isBF16())
    return false;
  return memref.getShape()[memref.getRank() - 1] ==
         vnni::utils::getVnniBlockingFactor(memref);
}

} // namespace utils
} // namespace vnni
} // namespace mlir
