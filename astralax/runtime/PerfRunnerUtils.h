//===- PerfRunnerUtils.h - Utils for debugging MLIR execution -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to measure performance (timers, statistics, etc).
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_EXECUTIONENGINE_PERFRUNNERUTILS_H
#define ASTL_EXECUTIONENGINE_PERFRUNNERUTILS_H

#include "mlir/ExecutionEngine/RunnerUtils.h"

//===----------------------------------------------------------------------===//
// Perf dialect utils
//===----------------------------------------------------------------------===//

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t perf_start_timer();

extern "C" MLIR_RUNNERUTILS_EXPORT double perf_stop_timer(int64_t);

#endif // ASTL_EXECUTIONENGINE_PERFRUNNERUTILS_H

