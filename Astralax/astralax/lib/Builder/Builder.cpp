#include "Builder/Builder.h"
#include "ONNX/OnnxConverter.h"

using namespace mlir;
namespace astl {
namespace hld {

ModuleOp ONNXBuilder::buildHLDMoudle(const std::string& name, const std::string& path) {
  auto unknownLoc = UnknownLoc::get(context);
  moduleOp = builder.create<ModuleOp>(unknownLoc, name);

  auto cvtr = std::make_shared<ONNXConverter>(moduleOp);
  cvtr->onnx2mlir(path);
  return moduleOp;
}

} // namespace hld
} // namespace astl