file(GLOB _srcs
     IR/*.cpp
     *.cpp)

add_mlir_library(ASTLDialect
    ${_srcs}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/astralax/include/Astralax/Dialect
    ${CMAKE_SOURCE_DIR}/build/astralax/include/Astralax/Dialect

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
)