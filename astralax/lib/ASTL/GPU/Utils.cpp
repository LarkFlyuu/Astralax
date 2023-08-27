//===- Utils.cpp -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/GPU/Utils.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

namespace mlir {
namespace astl {

void initializeGpuTargets() {
#ifdef ASTL_CUDA_ENABLE
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
#endif // ASTL_CUDA_ENABLE
}

} // namespace astl
} // namespace mlir
