add_mlir_dialect(XsmmOps xsmm)
add_mlir_doc(XsmmDialect XsmmDialect ASTL/ -gen-dialect-doc)
add_mlir_doc(XsmmOps XsmmOps ASTL/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS XsmmEnum.td)
mlir_tablegen(XsmmEnum.h.inc -gen-enum-decls)
mlir_tablegen(XsmmEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXsmmAttrDefIncGen)
