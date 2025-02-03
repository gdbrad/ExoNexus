## steps for projecting out correlators into correct irreps , result will be a new dataset in existing? h5 

open the existing hdf5 file (should be averaged over tsrc already at this point and )? so we just have a real vector of length 96(lattice temporal extent) containing the averaged correlator data 

extract the actual data from the cfg index

reshape the data into 96 rows. 
Rows: time slices 
columns: configurations 

compute the correlation time and other sanity checks: 
```
```
# This plots the first column, that will give a correlation function.
plot(m[, 1], main='First Row', xlab='Time Slice', ylab='Correlator')

# Then we plot the first time slice of all configurations, just because we can.
plot(m[1,], main='First Time Slice', xlab='Configuration Number', ylab='Correlator')

# We apply the Ulli Wolff method to compute the correlation time for fixed time
# slice accross the configurations.
corrs <- apply(m, 1, function(x) uwerrprimary(x))

corrs.val <- sapply(corrs, function(x) (x)$tauint)
corrs.err <- sapply(corrs, function(x) (x)$dtauint)

plotwitherror(1:length(corrs.val), corrs.val, corrs.err,
              main='Autocorrelation Across Configurations',
              xlab='Time Slice',
              ylab='Integrated Autocorrelation Time')
```



