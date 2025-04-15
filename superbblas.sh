#!/bin/bash

source config.sh

PKG=superbblas

#rm -rf $SOURCE_DIR/$PKG
rm -rf $BUILD_DIR/$PKG
rm -rf $INSTALL_DIR/$PKG

mkdir $BUILD_DIR/$PKG
mkdir $INSTALL_DIR/$PKG

pushd $SOURCE_DIR
git clone --branch master https://github.com/eromero-vlc/superbblas $PKG
popd

pushd $BUILD_DIR/$PKG
cp -r $SOURCE_DIR/$PKG/* .
export CFLAGS="$CFLAGS -fPIC"
export CXXFLAGS="$CXXFLAGS -fPIC"
make -j 16 install_cuda PREFIX=$INSTALL_DIR/$PKG 
popd
