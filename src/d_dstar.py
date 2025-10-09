"""
get lattice parameters from input file d_dstar.ini

"""

import numpy as np
import os
from opt_einsum import contract as oe_contract
import h5py
import argparse
import time
import yaml
import sys
from typing import Iterable, List, Dict
from gamma import gamma
import operator_factory as operator_factory
from operator_factory import QuantumNum
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
from contract_routines import *

# D:g5:disp_000:A1:mom_000

def parse_operator_name(operator:str,)

# create D meson insertion 
ins_D = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
# create D* insertion 
ins_Dstar = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)

# op_A = u_bar gamma5 c
op_D = Operator("D", [ins_D[0](0, 0, 0)], [1])
# op_B = c_bar gamma_i u
op_Ds = Operator("Dbar_star", [ins_Dstar[2](0, 0, 0)], [1])


# 
line_light = Propagator(perambulator_light, Lt)
line_charm = Propagator(perambulator_charm, Lt)
line_local_light = PropagatorLocal(perambulator_light, Lt)


D_src = Meson(elemental, op_D, True)
D_snk = Meson(elemental, op_D, False)
Ds_src = Meson(elemental, op_Ds, True)
Ds_snk = Meson(elemental, op_Ds, False)
