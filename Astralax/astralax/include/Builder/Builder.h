#ifndef ASTL_HLD_ONNX_BUILDER_H
#define ASTL_HLD_ONNX_BUILDER_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "Dialect/HLD/HLDOps.h"
#include "Utils/MLIRHelper.h"

using namespace mlir;
namespace astl {
namespace hld {
class ONNXBuilder {
public:
  ONNXBuilder(MLIRContext *context) : context(context), builder(context) {}
  virtual ~ONNXBuilder() = default;

  ModuleOp moduleOp;
  ModuleOp buildHLDMoudle(const std::string& path, const std::string& output);

  void save(const std::string& output) {
    outputCode(moduleOp, output);
  }
  void saveBinary(const std::string& output) {
    outputBinary(moduleOp, output);
  }
private:
  MLIRContext *context;
  OpBuilder builder;

};
} // namespace hld
} // namespace astl

#endif // ASTL_HLD_ONNX_BUILDER_H
