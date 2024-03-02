#ifndef ASTL_HELPER_H
#define ASTL_HELPER_H
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
void outputCode(const ModuleOp &module, const std::string &filenameWithExt);
void outputBinary(const ModuleOp &module, const std::string &output);
NameLoc getLoc(MLIRContext *context, const std::string &name);
}
#endif // ASTL_HELPER_H