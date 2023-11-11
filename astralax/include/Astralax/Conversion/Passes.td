#ifndef ASTL_ALL_PASSES
#define ASTL_ALL_PASSES

include "mlir/Pass/PassBase.td"

def Init : Pass<"init", "ModuleOp"> {
  let summary = "init module";
  let constructor = "createInitPass()";
  let dependentDialects = ["AstlDialect"];
}

#endif // ASTL_ALL_PASSES
