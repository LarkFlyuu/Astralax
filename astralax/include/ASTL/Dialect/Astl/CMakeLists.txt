add_mlir_dialect(AstlOps astl)
add_mlir_doc(AstlDialect AstlDialect ASTL/ -gen-dialect-doc)
add_mlir_doc(AstlOps AstlOps ASTL/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS AstlAttr.td)
mlir_tablegen(AstlAttr.h.inc -gen-enum-decls)
mlir_tablegen(AstlAttr.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAstlAttrDefIncGen)

set(LLVM_TARGET_DEFINITIONS AstlInterface.td)
mlir_tablegen(AstlInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(AstlInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAstlInterfaceIncGen)
add_dependencies(mlir-headers MLIRAstlInterfaceIncGen)
