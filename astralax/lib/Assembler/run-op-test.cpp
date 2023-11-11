#include "OpLib/Operator.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <string>
#include <iostream>
#include <dlfcn.h>

using namespace mlir;

void testOperator(std::string opPath, mlir::Operation *op) {
  void* handle = dlopen(opPath.c_str(), RTLD_LAZY);

  Operator* (*createAdd)();
  createAdd = (Operator* (*)())dlsym(handle, "create");
  Operator* addOp = createAdd();

  bool res = addOp->match(op);
  std::cout << " -> is match: " << res << std::endl;
  
  dlclose(handle);
}

int main(int argc, char **argv) {
  std::string opPath = argv[1];

  DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects(true);

  OpBuilder builder(&context);

  auto readModuleOp = parseSourceFile<ModuleOp>("linalg-conv2d.mlir", &context);
  
  /* pattern match */
  auto moduleOp = readModuleOp.get();
  Region &region = moduleOp.getRegion();
  for (auto &op : region.getBlocks().begin()->getOperations()) {
    std::cout << op.getName().getStringRef().str() << std::endl;
    testOperator(opPath, &op);
  }

  return 0;
}
