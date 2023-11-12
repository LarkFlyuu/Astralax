#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Astralax/InitAll.h"
#include "Astralax/Builder/Builder.h"

using namespace mlir;

int main(int argc, char **argv) {
  astl::registerAllPasses();

  DialectRegistry registry;
  astl::registerAllDialects(registry);

  std::string inputFile = argv[1];

  size_t len = inputFile.size();
  if (inputFile.substr(len - 5, len) == ".onnx") {
    MLIRContext context(registry);
    context.loadAllAvailableDialects();

    std::string moduleName;
    std::string onnxMLIRFile = "onnx.mlir";
    size_t pos = inputFile.find_last_of('/');
    if (pos != std::string::npos) {
      onnxMLIRFile = inputFile.substr(0, pos + 1) + onnxMLIRFile;
      moduleName = inputFile.substr(pos + 1, len - pos - 6);
    } else {
      moduleName = inputFile.substr(0, len - pos - 6);
    }

    auto astlBuilder = std::make_unique<astl::AstlBuilder>(&context);
    astlBuilder->buildAstlMoudle(moduleName, inputFile);
    astlBuilder->save(onnxMLIRFile);

    argv[1] = onnxMLIRFile.data();
  }

  return asMainReturnCode(
    MlirOptMain(argc, argv, "astralax optimizer driver\n", registry));
}
