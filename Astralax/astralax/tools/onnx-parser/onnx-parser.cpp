#include <iostream>

#include "Builder/Builder.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect,
                  astl::AstlDialect>();
  // return asMainReturnCode(
	// 	MlirOptMain(argc, argv, "XP MLIR module optimizer driver\n", registry,
	// 							/*preloadDialectsInContext=*/false));
	return 0;
}
