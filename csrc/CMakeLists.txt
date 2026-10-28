cmake_minimum_required(VERSION 3.20)
project(pinnjit LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(LLVM REQUIRED CONFIG HINTS /usr/lib/llvm-18/lib/cmake/llvm)
if(NOT LLVM_VERSION_MAJOR EQUAL 18)
  message(FATAL_ERROR "LLVM 18 required, found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

add_library(pinnjit SHARED codegen.cpp)
target_include_directories(pinnjit PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(pinnjit PRIVATE ${LLVM_DEFS_LIST})

if(LLVM_LINK_LLVM_DYLIB OR TARGET LLVM)
  target_link_libraries(pinnjit PRIVATE LLVM)
else()
  llvm_map_components_to_libnames(LLVM_LIBS core orcjit native nvptxcodegen nvptxinfo nvptxdesc passes)
  target_link_libraries(pinnjit PRIVATE ${LLVM_LIBS})
endif()

# Drop the library straight into the Python package.
set(PK_OUT ${CMAKE_CURRENT_SOURCE_DIR}/../pinn_kernels/_lib)
set_target_properties(pinnjit PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PK_OUT})
