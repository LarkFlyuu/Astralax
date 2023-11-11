#ifndef ASTL_ALL_PASSES
#define ASTL_ALL_PASSES

include "mlir/Pass/PassBase.td"

def Quant : Pass<"quant", "ModuleOp"> {
  let summary = "quantization pass";
  let constructor = "createQuantPass()";
  let dependentDialects = ["AstlDialect"];
}

#endif // ASTL_ALL_PASSES
