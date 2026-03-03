from https://web.mit.edu/~joshlin/www/dox/deriv__quark__displacement__w_8h_source.html#l00043

https://arxiv.org/pdf/hep-lat/0210030
http://arxiv.org/abs/0707.4162

## Running contractions on JURECA 
create a yaml input file see `a1mp.ini.yml` for an example. 

Run ``exotract.py --ini_file {INI_FILE} --check{Y/N}`` which will generate a batch script and create a new directory for that run , settting check will not launch the script. If not explicity given, the script will launch and then merge the output h5 files (if the run was successful). 

modify the slurm batch script see `slurm-test.sh` for example
`sbatch slurm-test.sh` 

2. we will now have a collection of h5 files which contain mesonic correlator data for a set of configurations (set to 5). these need to be merged into a single h5 file so run the script `python3 merge_h5.py` _this should be done automatically once all the slurm jobs complete_ 

3. now we need to project the correlators into the correct irrep , resulting in another h5 file WIP 

4. at this point the `exotraction` module has served its purpose. Head on over to `corrfit-exotics` for GEVP routines  

We employ the `opt_einsum' python library to perform contractions of meson elementals,perambulators,some gamma structure with covariant derivatives. 

Here we describe the necessary ingredients:
### Gamma conventions
Nothing groundbreaking here, just defining various gamma conventions in order to create different mesons eg. pseudoscalars and scalars. 

## derivative quark displacement

these "forward-backward" gauge covariant derivatives allow us to access states with higher angular momentum. On the latttice, these derivatives are finite displacements of quark fields connected by the gauge links. We will include zero, single, and double derivative operators. We must ensure that the operators have definite charge conjugation as well as a projection operator within the corresponding irreps.  

Apply "D_i" operator to the right onto source

   * $D_i = s_{ijk}\nabla_j\nabla_k$
   * where  $s_{ijk} = +1 \quad\forall i\ne j, j\ne k, i \ne k$
    return $F(z,0) D_\mu$

$$D_i = s_{ijk}\nabla_j\nabla_k$$
$$B_i = \epsilon_{ijk}\nabla_j\nabla_k$$
the symmetric lattice derivative is:
$$\nabla_\mu f(x) = U_\mu(x)f(x+\mu) - U_{-\mu}(x)f(x-\mu)$$

where
 
$$s_{ijk} = |\epsilon_{ijk}|$$
$$S_{\alpha jk} = 0\quad j\ne k, S_{111}=S_{222}=1, S_{122}=S_{233}=-1$$
we have set the displacement length(derivative length)to 1 in our chroma input files

in our chroma input files for the meson elementals,which corresponds to the displacement length in derivative
```
disp                     Group
disp_1                   Group
disp_1_1                 Group
disp_1_2                 Group
disp_1_3                 Group
disp_2                   Group
disp_2_1                 Group
disp_2_2                 Group
disp_2_3                 Group
disp_3                   Group
disp_3_1                 Group
disp_3_2                 Group
disp_3_3                 Group
```
* construct the (right Nabla) source 
$\nabla_\mu f(x) = U_\mu(x)f(x+\mu) - U_{-\mu}(x)f(x-\mu)$

$\nabla_\mu F(x)$
    The sink interpolator structure is
    $\Gamma_f \equiv \nabla_i$

* Construct (right D) source

    Operator is  D
    The sink interpolator structure is
    $\Gamma_f \equiv D_i$

* Construct (right B) source

    Operator is  B
    The sink interpolator structure is
    $\Gamma_f \equiv B_i$

* Construct (right E) source

    Operator is  E
    The sink interpolator structure is
    $\Gamma_f \equiv E_i$

* Construct (right Laplacian) source
       Operator is  Laplacian
      The sink interpolator structure is
      $\Gamma_f \equiv Laplacian$
## pion operators 

* Construct (PionxNabla_T1) source
     
    Operator is  Pion x Nabla_T1
    The sink interpolator structure is
    $\Gamma_f \equiv \gamma_5\nabla_i$

* Construct (PionxD_T2) source
     
    Operator is  pion x D_T2
    The sink interpolator is   
    $\Gamma_f \equiv \gamma_4\gamma_5 D_i$  

* Construct 5 source

    Operator is  pion x B_T1
    The sink interpolator is   

    $\Gamma_f \equiv \gamma_5 B_i$  

## mixed gamma operators 

build correlation matrix with multiple gamma operators 

build a 2x2 correlation matrix with both operators gamma_i, nabla_i

derivative based gamma operators result in elementals no longer diagonal in distillation space 

## pion operators:
pionxNABLA_T1
pion_2xNABLA_T1
pionxD_T2
pion_2xD_T2
pionxB_T1
pion_2xB_T1

The pion with $J^{PC}=0^{-}$ with zero total momentum trivially subduces into the 1D $A_1^{-}$ irrep which has a subduction coeff of 1, thus nothing needs to be done internally within the contractions. 

CHECK THIS: operators at rest reside in irreps of double-covers of the octahedral group with parity, defined as $O_h^D$. 

### pion in flight 
00n $-\in Dic_4$
0nn
nnn
nm0
nnm 


full list of operators at the source: 

("pionxNABLA_T1", "a0xNABLA_T1", "a0_2xNABLA_T1", "pion_2xNABLA_T1", 
		     "rhoxNABLA_A1", "rhoxNABLA_T1", "rhoxNABLA_T2", "rhoxNABLA_E", 
		     "rho_2xNABLA_A1", "rho_2xNABLA_T1", "rho_2xNABLA_T2", "rho_2xNABLA_E", 
		     "a1xNABLA_A1", "a1xNABLA_T1", "a1xNABLA_T2", "a1xNABLA_E", 
		     "b1xNABLA_A1", "b1xNABLA_T1", "b1xNABLA_T2", "b1xNABLA_E", 
		     "pionxD_T2", "a0xD_T2", "a0_2xD_T2", "pion_2xD_T2",
		     "a1xD_A2", "a1xD_T1", "a1xD_T2", "a1xD_E", 
		     "b1xD_A2", "b1xD_T1", "b1xD_T2", "b1xD_E", 
		     "rhoxD_A2", "rhoxD_T1", "rhoxD_T2", "rhoxD_E", 
		     "rho_2xD_A2", "rho_2xD_T1", "rho_2xD_T2", "rho_2xD_E", 
		     "pionxB_T1", "a0xB_T1", "a0_2xB_T1", "pion_2xB_T1", 
		     "rhoxB_A1", "rhoxB_T1", "rhoxB_T2", "rhoxB_E", 
		     "rho_2xB_A1", "rho_2xB_T1", "rho_2xB_T2", "rho_2xB_E", 
		     "a1xB_A1", "a1xB_T1", "a1xB_T2", "a1xB_E",
		     "b1xB_A1", "b1xB_T1", "b1xB_T2", "b1xB_E")

## projected operators 

In order to select the irreps commensurate with the given momentum, P^2, and "project out" existing two-point correlators into the correct irrep, we have to __extract projected operator coefficients__ 

How JLAB does it in ``nstar`` is to read in all of the ops and build a big map. 

So we need to: 
```
for ir in irreps: 
    get all possible states in irrep 
    extract projected operator for state in states in irrep 
