//===- Utils.h - GPU-related helpers --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_GPU_UTILS_H
#define ASTL_GPU_UTILS_H

namespace mlir {
namespace astl {

void initializeGpuTargets();

} // namespace astl
} // namespace mlir

#endif // ASTL_GPU_UTILS_H
