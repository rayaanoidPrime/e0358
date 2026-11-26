# e0358-tile: affine matmul nests to CUDA Tile IR, built against the MLIR
# build (MLIR_DIR) and the cuda-tile build (CUDA_TILE_DIR, the source tree
# with the build in CUDA_TILE_DIR/build).
cmake_minimum_required(VERSION 3.20)
project(e0358-tile LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})
include(AddLLVM)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${CUDA_TILE_DIR}/include ${CUDA_TILE_DIR}/build/include)
link_directories(${CUDA_TILE_DIR}/build/lib)

add_llvm_executable(e0358-tile e0358-tile.cpp)
target_link_libraries(e0358-tile PRIVATE
  CudaTileBytecodeWriter CudaTileBytecodeCommon CudaTileDialect
  MLIRAffineDialect MLIRArithDialect MLIRFuncDialect MLIRMemRefDialect
  MLIRParser)
