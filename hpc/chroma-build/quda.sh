#!/bin/bash

source config.sh

PKG=quda

#rm -rf $SOURCE_DIR/$PKG
rm -rf $BUILD_DIR/$PKG
rm -rf $INSTALL_DIR/$PKG

mkdir $BUILD_DIR/$PKG
mkdir $INSTALL_DIR/$PKG

pushd $SOURCE_DIR
git clone --branch develop https://github.com/lattice/quda.git $PKG
popd

pushd $BUILD_DIR/$PKG
$CMAKE -DCMAKE_CXX_FLAGS="-std=c++20 -march=native -fopenmp -fPIC -O2"      \
       -DCMAKE_BUILD_TYPE="RELEASE"                                         \
       -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/$PKG                             \
       -DQMP_DIR=$INSTALL_DIR/qmp/lib/cmake/QMP                             \
       -DQIO_DIR=$INSTALL_DIR/qdpjit/lib/cmake/QIO                          \
       -DQDPXX_DIR=$INSTALL_DIR/qdpjit/lib/cmake/QDPXX                      \
       -DLLVM_DIR=$INSTALL_DIR/llvm/lib/cmake/llvm                          \
       -DQUDA_GPU_ARCH=$GPU_ARCH                                            \
       -DQUDA_BUILD_SHAREDLIB=ON                                            \
       -DQUDA_MPI=OFF                                                       \
       -DQUDA_QMP=ON                                                        \
       -DQUDA_QIO=ON                                                        \
       -DQUDA_QDPJIT=ON                                                     \
       -DQUDA_INTERFACE_QDPJIT=ON                                           \
       -DQUDA_CLOVER_DYNAMIC=OFF                                            \
       -DQUDA_CLOVER_RECONSTRUCT=OFF                                        \
       -DQUDA_MULTIGRID=ON                                                  \
       -DQUDA_CONTRACT=ON                                                   \
       -DQUDA_COVDEV=ON                                                     \
       -DQUDA_DIRAC_DEFAULT=OFF                                             \
       -DQUDA_DIRAC_DEFAULT_OFF=ON                                          \
       -DQUDA_DIRAC_WILSON=ON                                               \
       -DQUDA_DIRAC_CLOVER=ON                                               \
       -DQUDA_BUILD_ALL_TESTS=OFF                                           \
       $CMAKE_EXTRA_FLAGS                                                   \
       $SOURCE_DIR/$PKG
$CMAKE --build . $CMAKE_MAKE_FLAGS
$CMAKE --install .
popd


