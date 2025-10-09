## $DD^*$ contractions
I think we can come up with a better name than redstar...

As we will attempt to reverse-engineer the functionality of the `redstar` package from JLAB, we will follow the naming conventions in https://github.com/gdbrad/chroma-scripts-cori/blob/master/ensembles.sh 

Substrings dictionary -> operator String:
To populate our string representing a particular contraction -> 2pt correlator, we will concatenate the following into an operator string:

- momentum letters: the momentum label 100 corresponds to all orientations of (1,0,0).
When we contract the perambulator and meson elementalss, we must average over all possible orientations.
    args: 3 
    Options:

    Group | $\Lambda^c$ (dim) | # ops
    000   | 
    n00   | Di_c4
    nn0   | Di_c2
    nm0
    nnn   | Di_c4
    nnm
    nmk
- operators: 
    - $D(0)D^*(0)$ 
    - $D(1)D^*(1)$
    - $D^*(0)D^*(0)$  


# Rest-frame systems : Zero-momentum operators(mesons at rest) P_tot = 0,0,0:
All seven conjugacy classes of little groups can serve as stabilizer subgroups within the total symmetry group, which is $O_h$. 
- $H_P^{(s)} = O_h: [\textbf{n}] = [(0,0,0)]$ 


Irreps: $T_1^+$ , $A_1^-$ 
## nsq = 0
|DD_Iz2_A1_000_000_000> = |0,0,0>|0,0,0>

# nsq = 1
|DD_Iz2_A1_000_100_100> = 1./sqrt(6) * (
    + |1,0,0>|-1,0,0> + |0,1,0>|0,-1,0> + |0,0,1>|0,0,-1>
    + |-1,0,0>|1,0,0> + |0,-1,0>|0,1,0> + |0,0,-1>|0,0,1>
    )

# nsq = 2
|DD_Iz2_A1_000_110_110> = 1./(2*sqrt(3)) * (
    + |1,1,0>|-1,-1,0> + |0,1,1>|0,-1,-1> + |1,0,1>|-1,0,-1> + |1,-1,0>|-1,1,0>
    + |0,1,-1>|0,-1,1> + |-1,0,1>|1,0,-1> + |-1,1,0>|1,-1,0> + |0,-1,1>|0,1,-1>
    + |1,0,-1>|-1,0,1> + |-1,-1,0>|1,1,0> + |0,-1,-1>|0,1,1> + |-1,0,-1>|1,0,1>
    )

# nsq = 3
|DD_Iz2_A1_000_111_111> = 1./(2*sqrt(2)) * (
    + |1,1,1>|-1,-1,-1> + |-1,1,1>|1,-1,-1> + |1,-1,1>|-1,1,-1> + |1,1,-1>|-1,-1,1>
    + |-1,-1,1>|1,1,-1> + |1,-1,-1>|-1,1,1> + |-1,1,-1>|1,-1,1> + |-1,-1,-1>|1,1,1>
    )

# nsq = 4
|DD_Iz2_A1_000_200_200> = 1./sqrt(6) * (
    + |2,0,0>|-2,0,0> + |0,2,0>|0,-2,0> + |0,0,2>|0,0,-2>
    + |-2,0,0>|2,0,0> + |0,-2,0>|0,2,0> + |0,0,-2>|0,0,2>
    )

T_000="DD_MG1g1MxD0J0S_J1o2_G1g1"

## Non-zero momentum operators (mesons in flight):
	T_n00="DD_MG1g1MxD0J0S_J1o2_H1o2D4E1"
	T_nn0="DD_MG1g1MxD0J0S_J1o2_H1o2D2E"
	T_nnn="DD_MG1g1MxD0J0S_J1o2_H1o2D3E1"
	T_nm0="DD_MG1g1MxD0J0S_J1o2_H1o2C4nm0E"
	T_nnm="DD_MG1g1MxD0J0S_J1o2_H1o2C4nnmE"
	redstar_insertioexotictors="\
pion_pionxDX__J0_A1
pion_pion_2xDX__J0_A1
rho_rhoxDX__J1_T1
rho_rho_2xDX__J1_T1
b_b1xDX__J1_T1
b_b0xDX__J0_A1
a_a1xDX__J1_T1
a_a0xDX__J0_A1
```

phi1_phi2_Iz_A1_Ptot_p1_p2
eg
pion_kaon_Iz32_A1_100_100_000



## $DD^*$ 

# Inelastic threshold begins at E = 4mpi
# E = 2 * mpi * sqrt( 1 + nsq * (2 * pi / L / mpi)**2 )
# sqrt( 1 + nsq * (2 * pi / L / mpi)**2 ) = 2
# nsq = 3 * (mpi * L / 2 / pi)**2
# for mpi * L = 4, this occurs for nsq = 1.2



# Ptot = 1,0,0
# 6 states in all shells
# nsq = 0
|DD_Iz2_A1_100_nsq0_0> = |000>|100>
|DD_Iz2_A1_100_nsq0_1> = |000>|-100>
|DD_Iz2_A1_100_nsq0_2> = |000>|010>
|DD_Iz2_A1_100_nsq0_3> = |000>|0-10>
|DD_Iz2_A1_100_nsq0_4> = |000>|001>
|DD_Iz2_A1_100_nsq0_5> = |000>|00-1>

# nsq = 1
# 0,0,1
|DD_Iz2_A1_100_100_101,0> = 0.5 * (
    |1,0,0>|-1,0,1> + |0,1,0>|0,-1,1> + |-1,0,0>|1,0,1> + |0,-1,0>|0,1,1>
    )
# 0,0,-1
|DD_Iz2_A1_100_100_101,1> = 0.5 * (
    )
# 0,1,0
|DD_Iz2_A1_100_100_101,2> = 0.5 * (
    |1,0,0>|-1,1,0> + |0,0,1>|0,1,-1> + |-1,0,0>|1,1,0> + |0,0,-1>|0,1,1>
    )
# 0,-1,0
|DD_Iz2_A1_100_100_101,3> = 0.5 * (
    )
# 1,0,0
|DD_Iz2_A1_100_100_101,4> = 0.5 * (
    |0,1,0>|1,-1,0> + |0,0,1>|1,0,-1> + |0,-1,0>|1,1,0> + |0,0,-1>|1,0,1>
    )
# -1,0,0
|DD_Iz2_A1_100_100_101,5> = 0.5 * (
    |0,1,0>|-1,-1,0> + |0,0,1>|-1,0,-1> + |0,-1,0>|-1,1,0> + |0,0,-1>|-1,0,1>
    )
