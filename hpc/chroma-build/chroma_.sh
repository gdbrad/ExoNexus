#!/bin/bash
source config.sh

PKG=chroma

#rm -rf $SOURCE_DIR/$PKG
rm -rf $BUILD_DIR/$PKG
rm -rf $INSTALL_DIR/$PKG

mkdir $BUILD_DIR/$PKG
mkdir $INSTALL_DIR/$PKG

#pushd $SOURCE_DIR
#git clone --recursive --branch devel https://github.com/JeffersonLab/chroma.git $PKG
#popd

pushd $BUILD_DIR/$PKG
$CMAKE -DCMAKE_C_FLAGS="-march=native -fopenmp -fPIC -O2 -DH5_USE_110_API"          \
       -DCMAKE_CXX_FLAGS="-std=c++20 -fconcepts -march=native -fopenmp -fPIC -O2 -DH5_USE_110_API -lhdf5"   \
       -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/$PKG                                     \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo                                            \
       -DBUILD_SHARED_LIBS=ON                                                       \
       -DQUDA_DIR=$INSTALL_DIR/quda/lib/cmake/QUDA                                  \
       -DQDPXX_DIR=$INSTALL_DIR/qdpjit/lib/cmake/QDPXX                              \
       -DQMP_DIR=$INSTALL_DIR/qmp/lib/cmake/QMP                                     \
       -DLLVM_DIR=$INSTALL_DIR/llvm/lib/cmake/llvm                                  \
       -DPRIMME_DIR=$INSTALL_DIR/primme                                             \
       -DSUPERBBLAS_DIR=$INSTALL_DIR/superbblas                                     \
       -DChroma_ENABLE_JIT_CLOVER=ON                                                \
       -DChroma_ENABLE_OPENMP=ON                                                    \
       -DChroma_ENABLE_SSE2=ON                                                      \
        -DChroma_ENABLE_SSE3=ON                                                      \
        -DChroma_ENABLE_QUDA=ON                                                      \
        -DChroma_ENABLE_SUPERBBLAS=ON                                                \
        -DChroma_ENABLE_PRIMME=ON                                                    \
        $CMAKE_EXTRA_FLAGS                                                           \
        $SOURCE_DIR/$PKG
$CMAKE --build . --parallel 24
$CMAKE --install .
popd