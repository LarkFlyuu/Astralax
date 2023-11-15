macro(custom_protobuf_find)
  message(STATUS "Use custom protobuf build.")
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_BUILD_EXAMPLES "" OFF)
  option(protobuf_WITH_ZLIB "" OFF)
  option(protobuf_BUILD_SHARED_LIBS "" ${BUILD_SHARED_LIBS})
  
  set(UBSAN_FLAG "-fsanitize=undefined")

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/protobuf)
  
  if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf ALIAS libprotobuf)
    add_library(protobuf::libprotobuf-lite ALIAS libprotobuf-lite)
    
    if(NOT (ANDROID OR IOS))
      add_executable(protobuf::protoc ALIAS protoc)
    endif()
  endif()
endmacro()

custom_protobuf_find()

set(XPENGRT_PROTOC_EXECUTABLE protobuf::protoc)