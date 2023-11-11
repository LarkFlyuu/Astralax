#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Astralax/InitAll.h"

using namespace mlir;

int main(int argc, char **argv) {
  astl::registerAllPasses();

  DialectRegistry registry;
  astl::registerAllDialects(registry);

  return asMainReturnCode(
    MlirOptMain(argc, argv, "astralax optimizer driver\n", registry));
}
