#!/bin/bash

source config.sh

PKG=qdpjit

#rm -rf $SOURCE_DIR/$PKG
rm -rf $BUILD_DIR/$PKG
rm -rf $INSTALL_DIR/$PKG

mkdir $BUILD_DIR/$PKG
mkdir $INSTALL_DIR/$PKG

pushd $SOURCE_DIR
git clone --recursive --branch devel https://github.com/JeffersonLab/qdp-jit.git $PKG
popd

#cp 0001-hdf5-fix.patch $SOURCE_DIR/$PKG

pushd $SOURCE_DIR/$PKG
#git apply 0001-hdf5-fix.patch
popd

pushd $BUILD_DIR/$PKG
$CMAKE -DCMAKE_C_FLAGS="-march=native -fopenmp -fPIC -O2 -DH5_USE_110_API"        \
       -DCMAKE_CXX_FLAGS="-march=native -fopenmp -fPIC -O2 -DH5_USE_110_API"      \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo                                          \
       -DQDP_ENABLE_BACKEND=CUDA                                                  \
       -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/$PKG                                   \
       -DQMP_DIR=$INSTALL_DIR/qmp/lib/cmake/QMP                                   \
       -DLLVM_DIR=$INSTALL_DIR/llvm/lib/cmake/llvm                                \
       -DQDP_ENABLE_LLVM16=ON                                                     \
       -DQDP_ENABLE_COMM_SPLIT_DEVICEINIT=ON                                      \
       -DBUILD_SHARED_LIBS=ON                                                     \
       -DQDP_PROP_OPT=OFF                                                         \
       -DQIO_ENABLE_PARALLEL_IO=ON                                                \
       -DQDP_PARALLEL_ARCH=parscalar                                              \
       -DQDP_PRECISION=double                                                     \
       -DQDP_BUILD_EXAMPLES=OFF                                                   \
       -DQDP_ENABLE_MANAGED_MEMORY=ON                                             \
       -DQDP_USE_HDF5=ON                                                          \
       $CMAKE_EXTRA_FLAGS                                                         \
       $SOURCE_DIR/$PKG
$CMAKE --build . $CMAKE_MAKE_FLAGS
$CMAKE --install .
popd
