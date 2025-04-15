#!/bin/bash

source config.sh

PKG=primme

#rm -rf $SOURCE_DIR/$PKG
rm -rf $BUILD_DIR/$PKG
rm -rf $INSTALL_DIR/$PKG

mkdir $BUILD_DIR/$PKG
mkdir $INSTALL_DIR/$PKG

pushd $SOURCE_DIR
git clone --branch eloy/primme-3.3 https://github.com/primme/primme  $PKG
popd

pushd $BUILD_DIR/$PKG
cp -r $SOURCE_DIR/$PKG/* .
export CFLAGS="$CFLAGS -fPIC"
make -j 16 lib install PREFIX=$INSTALL_DIR/$PKG
cp lib/libprimme.a $INSTALL_DIR/$PKG/lib
popd
