add_mlir_library(astl_c_runner_utils
  SHARED
  XsmmRunnerUtils.cpp
  PerfRunnerUtils.cpp

  LINK_LIBS PUBLIC
  xsmm
)
set_property(TARGET astl_c_runner_utils PROPERTY CXX_STANDARD 11)
target_compile_definitions(astl_c_runner_utils PRIVATE mlir_c_runner_utils_EXPORTS)