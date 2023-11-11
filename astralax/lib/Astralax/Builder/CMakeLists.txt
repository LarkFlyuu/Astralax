file(GLOB _srcs *.cpp)

add_mlir_library(ASTLBuilder
    ${_srcs}

    DEPENDS
    MLIRAstlOpsIncGen
    MLIRAstlAttrIncGen
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    ASTLDialect
)