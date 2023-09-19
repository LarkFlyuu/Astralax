add_subdirectory(OpLib)

add_executable(run-op-test run-op-test.cpp)

target_link_libraries(
  run-op-test 
  
  dl ASTLGPU
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
