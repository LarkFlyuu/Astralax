#include "Astralax/Builder/Builder.h"
#include "ONNX/OnnxConverter.h"

using namespace mlir;
namespace astl {

void AstlBuilder::buildAstlMoudle(const std::string& name, const std::string& path) {
  auto unknownLoc = UnknownLoc::get(context);
  moduleOp = builder.create<ModuleOp>(unknownLoc, name);

  auto cvtr = std::make_shared<ONNXConverter>(moduleOp);
  cvtr->onnx2mlir(path);
}

};