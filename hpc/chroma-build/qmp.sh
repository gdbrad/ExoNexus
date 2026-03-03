#!/bin/bash

source config.sh

PKG=qmp

#rm -rf $SOURCE_DIR/$PKG
rm -rf $BUILD_DIR/$PKG
rm -rf $INSTALL_DIR/$PKG

mkdir $BUILD_DIR/$PKG
mkdir $INSTALL_DIR/$PKG

pushd $SOURCE_DIR
git clone --branch master https://github.com/usqcd-software/qmp.git $PKG
popd

pushd $BUILD_DIR/$PKG
$CMAKE                                            \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo                  \
       -DCMAKE_C_FLAGS="-march=native -fopenmp -fPIC -O2 "\
       -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/$PKG           \
       -DQMP_MPI=ON                                       \
       -DBUILD_SHARED_LIBS=ON                             \
       $CMAKE_EXTRA_FLAGS                                 \
       $SOURCE_DIR/$PKG
$CMAKE --build . $CMAKE_MAKE_FLAGS
$CMAKE --install .
popd
