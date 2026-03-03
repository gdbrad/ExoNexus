lqcd analysis library to extract invariant amplitudes for single and dimeson systems. These amplitudes contribute to the helicity-dependent matrix elements. 

## features
- statistical resampling 
tsrc averaging 
ensemble average over all gauge field cfgs 
jackknife 
bootstrap

### jackknife resmapling
this fills entries of jackknifed array starting with the "gth" gauge config. 
then we compute the ensemble average for each jackknife sample. 
then compute data covariance and the inverse 
so essentially take the difference (for both real and imag components) bettween the jackknife average at index g and time index ti and the ensemble avg at time inx ti then multiply by the same but at time idx tj. 

so now we have the data covariance we can compute its inverse. 
i think we ignore the t=0 cntact term? 

we need to correct for bias when forming ratios of corr cunctions for EACH jackknife ensemble average. 

we need to form collections of means and bins as datasets? i think gvar does this natively. 
 
- bayesian fitting with lsqfit and gvar to fit 2pt functions 
- gevp routines to compute principal correlators 
- fits to principal correlators 


now onto extracting the amplitudes from these fits. 

Pull out amplitudes extracted per jackknife sample and organize data into arrays


### dispersion relatio 
