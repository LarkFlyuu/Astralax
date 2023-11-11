file(GLOB _srcs
          Init/*.cpp
)

add_mlir_library(ASTLConversion
    ${_srcs}
    
    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/astralax/include/Astralax
    ${CMAKE_SOURCE_DIR}/build/astralax/include/Astralax

    DEPENDS
    ASTLConversionIncGen
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
)