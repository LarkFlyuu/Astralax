add_mlir_dialect(AstlOps astl)

set(LLVM_TARGET_DEFINITIONS AstlOps.td)
mlir_tablegen(AstlEnums.h.inc -gen-enum-decls)
mlir_tablegen(AstlEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(AstlAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(AstlAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAstlAttrIncGen)
