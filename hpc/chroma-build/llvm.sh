#!/bin/bash

source config.sh

PKG=llvm

#rm -rf $SOURCE_DIR/$PKG
rm -rf $BUILD_DIR/$PKG
rm -rf $INSTALL_DIR/$PKG

mkdir $BUILD_DIR/$PKG
mkdir $INSTALL_DIR/$PKG

pushd $SOURCE_DIR
git clone --depth 1 --branch llvmorg-16.0.6 https://github.com/llvm/llvm-project.git $PKG
popd

pushd $BUILD_DIR/$PKG
$CMAKE                                       \
       -DBUILD_SHARED_LIBS="ON"                        \
       -DLLVM_ENABLE_PROJECTS="llvm"                    \
       -DLLVM_ENABLE_TERMINFO="OFF"                     \
       -DLLVM_ENABLE_ZLIB="OFF"                         \
       -DLLVM_ENABLE_RTTI="ON"                          \
       -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/$PKG         \
       -DLLVM_TARGETS_TO_BUILD="X86;NVPTX"              \
       -DCMAKE_BUILD_TYPE="Release"                     \
       $CMAKE_EXTRA_FLAGS                               \
       $SOURCE_DIR/$PKG/llvm
$CMAKE --build . $CMAKE_MAKE_FLAGS
$CMAKE --install .
popd
