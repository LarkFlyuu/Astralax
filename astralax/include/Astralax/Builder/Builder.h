#ifndef __ASTL_MLIR_BUILDER__
#define __ASTL_MLIR_BUILDER__
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "Astralax/Dialect/AstlOps.h"
#include "Astralax/Helper/Helper.h"

using namespace mlir;
namespace astl {
class AstlBuilder {
public:
  AstlBuilder(MLIRContext *context) : context(context), builder(context) {}
  virtual ~AstlBuilder() = default;

  ModuleOp moduleOp;
  void buildAstlMoudle(const std::string& path, const std::string& output);
  void save(const std::string& output) {
    outputCode(moduleOp, output);
  }
  void saveBinary(const std::string& output) {
    outputBinary(moduleOp, output);
  }

private:
  MLIRContext *context;
  OpBuilder builder;

  void createAstlConstantOp();
  void createAstlConvOp();
  void createAstlDeConvOp();
  void createAstlFCOp();

};
} // namespace astl

#endif // __ASTL_MLIR_BUILDER__