#ifndef __ASTL_MLIR_BUILDER__
#define __ASTL_MLIR_BUILDER__
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "Astralax/Dialect/AstlOps.h"

using namespace mlir;
namespace astl {
class AstlBuilder {
public:
  AstlBuilder(MLIRContext *context) : context(context), builder(context) {
    unknownLoc = UnknownLoc::get(context);
  }
  virtual ~AstlBuilder() = default;

  void buildAstlMoudle(const std::string& name, const std::string& path);

private:
  MLIRContext *context;
  OpBuilder builder;
  Location unknownLoc{nullptr};

  void setInsertionPointToStart(Block *block) {
    builder.setInsertionPointToStart(block);
  }
  void setInsertionPointToEnd(Block *block) {
    builder.setInsertionPointToEnd(block);
  }
  void setInsertionPoint(Operation *op) {
    builder.setInsertionPoint(op);
  }
  void setInsertionPointAfter(Operation *op) {
    builder.setInsertionPointAfter(op);
  }

  void createAstlConstantOp();
  void createAstlConvOp();
  void createAstlDeConvOp();
  void createAstlFCOp();

};
} // namespace astl

#endif // __ASTL_MLIR_BUILDER__