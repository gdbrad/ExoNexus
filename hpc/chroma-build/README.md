# chroma-jureca-build

## Order of operations

1. Add your env file (if using jureca then `env-new-jureca-gpu.sh` should suffice) for module loading and CMAKE directives
2. Modify `config.sh` to source your env file and the base directory for installation  
3. `./superbblas.sh`
4. `./primme.sh`
5. `./llvm.sh`
6. `./qmp.sh`
7. `./qdpjit.sh`
8. `./quda.sh`
9. `./chroma.sh`

In each of the above files, the following lines must be uncommented for a new build 
```
#rm -rf $SOURCE_DIR/$PKG
#rm -rf $BUILD_DIR/$PKG
#rm -rf $INSTALL_DIR/$PKG

#mkdir $BUILD_DIR/$PKG
#mkdir $INSTALL_DIR/$PKG

#pushd $SOURCE_DIR
#git clone --recursive --branch devel https://github.com/JeffersonLab/chroma.git $PKG
#popd
```

## SDB -> H5 Conversion scripts
If one needs to convert the output chroma binaries to HDF5 manually (There is H5 conversion capability in https://github.com/callat-qcd/lalibe but we have not implemented this)

See https://github.com/gdbrad/chroma_sdb-h5-convert  

The files `convert_meson96.cc` and `convert_peram96.cc` need to be added to the directory `{PATH-TO-CHROMA-DIR}/source/chroma/mainprogs/main`. This can be done after building `Chroma`, just comment out everything in `chroma.sh` except 
```
source config.sh
PKG=chroma
pushd $BUILD_DIR/$PKG
$CMAKE --build . --parallel 24
$CMAKE --install .
popd
```
and re-run `./chroma.sh` 

Make sure to add the conversion binaries to the CMakeLists.txt file in {PATH-TO-CHROMA-DIR}/source/chroma/mainprogs/main/CMakeLists.txt; 
```
set( ChromaMain_EXECUTABLES 
	chroma
	purgaug
	cfgtransf
	const_hmc
	hmc
	spectrum_s
    collect_propcomp 
    qpropgfix qproptrev 
    qpropqio  
    qproptransf 
    wallformfac
   convert_peram96
  convert_meson96 
)
```
AND 
```
INSTALL( TARGETS hmc const_hmc chroma purgaug cfgtransf convert_peram96 convert_meson96 
		 EXPORT ChromaTargets
		 ARCHIVE DESTINATION lib
 		 LIBRARY  DESTINATION lib
  		 RUNTIME DESTINATION bin
  		 INCLUDES DESTINATION include
```