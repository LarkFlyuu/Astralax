file(GLOB _srcs *.cpp)

add_subdirectory(Interfaces)

add_mlir_library(HldDialect
    ${_srcs}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/astralax/include/Astralax/Dialect
    ${CMAKE_SOURCE_DIR}/build/astralax/include/Astralax/Dialect

    DEPENDS
    MLIRHldOpsIncGen
    MLIRHldAttrIncGen
    MLIRShapeInterfaceIncGen
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    MLIRShapeInterface
)