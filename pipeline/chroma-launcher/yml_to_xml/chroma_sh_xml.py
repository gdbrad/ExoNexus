'''generate xmls and spawn a chroma instance '''

import os 
from pydantic import BaseModel,Field
#coding: utf-8
from typing import Dict,Any,Optional
from decimal import Decimal
import re 

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
print(PROJECT_DIR)
FDIR = os.path.dirname(os.path.realpath(__file__))
INFILES = os.path.abspath(os.path.join(FDIR, "ens_files"))
TEMPLATE = os.path.join(FDIR,'templates')
OUTPATH = os.path.join(FDIR,'res/slurm_scripts')
# ENSEMBLES = 

# DATA = os.path.abspath(os.path.join(FDIR, os.pardir, "cfgs"))
# files = os.listdir(CONFS_PATH)
# for file in files:
#     data = [file.rstrip(".lime")]
#     print(data)

import re
from typing import Dict, Any

def parse_ensemble(short_tag: str) -> Dict[str, Any]:
    """should distinguish erics and gios ensemble naming conventions"""
    # Ensemble definitions
    long_tag = {
    "a125m280": "b3.30_ms-0.057_mud-0.129_s32t64-000-0001-0400",
    "eric_test": "b3.70_ms0.000_mud-0.022_s32t96-000",
    "eric_s32t64": "b3.6_mc0.25_mud-0.013_s32t64",
    "eric_s40t64": "b3.6_mc0.25_mud-0.013_s40t64",
    "eric_s48t64": "b3.6_mc0.25_mud-0.013_s48t64",
    "a065m420": "b3.70_ms0.000_mud-0.0200_s32t96-000",
    "a065m380": "b3.70_ms0.000_mud-0.0220_s32t96-000",
    "a065m300": "b3.70_ms0.000_mud-0.0250_s40t96-000",
    "a085m420": "b3.57_ms-0.007_mud-0.0380_s24t64-000",
    "a085m300": "b3.57_ms-0.007_mud-0.0440_s32t64-000",
    "a085m200": "b3.57_ms-0.007_mud-0.0483_s48t64-000",
    "a125m400": "b3.30_ms-0.057_mud-0.1200_s16t64-000",
    "a125m330": "b3.30_ms-0.057_mud-0.1233_s24t64-000",
    "a125m280": "b3.30_ms-0.057_mud-0.1265_s24t64-000",
    }
    key = long_tag[short_tag]

    patterns = [
        # standard format with ms and mud
        r"b(?P<beta>[0-9]+\.[0-9]+)"
        r"_ms(?P<ms>-?[0-9]+\.[0-9]{3})"
        r"_mud-(?P<mud>[0-9]+\.[0-9]{4})"
        r"_s(?P<NL>[0-9]{2})"
        r"t(?P<NT>[0-9]{2})"
        r"-(?P<P>[0-9]{3})",

        # Pattern for s32t64 format with mc instead of ms
        r"b(?P<beta>[0-9]+\.[0-9]+)"
        r"_mc(?P<mc>[0-9]+\.[0-9]{2})"
        r"_mud-(?P<mud>[0-9]+\.[0-9]{3})"
        r"_s(?P<NL>[0-9]{2})"
        r"t(?P<NT>[0-9]{2})",

        # minimal
        r"b(?P<beta>[0-9]+\.[0-9]+)"
        r"_ms(?P<ms>-?[0-9]+\.[0-9]{3})"
        r"_mud-(?P<mud>[0-9]+\.[0-9]{3})"
        r"_s(?P<NL>[0-9]{2})"
        r"t(?P<NT>[0-9]{2})"
    ]

    # Type mapping for conversion
    type_map = {
        "beta": str,
        "NT": int,
        "NL": int,
        "P": str,
        "mud": str,
        "ms": str,
        "mc": str,
    }

    info = {}
    for pattern in patterns:
        match = re.match(pattern, key)
        if match:
            groups = match.groupdict()
            info = {
                key: type_map[key](val)
                for key, val in groups.items()
                if key in type_map and val is not None
            }
            break

    return info

# Test cases
# for tag in ["a125m280", "test", "s32t64"]:
#     ens_props = parse_ensemble(tag)
#     print(f"{tag}: {ens_props}")
# ens_props = parse_ensemble("s40t64")

class ChromaOptions(BaseModel):
    '''
    load chroma run options from yaml file to pass to launcher shell script 
    last step in sequence prior to running chroma job on cluster
    '''
    user: str
    ens_short: str 
    P: int 
    cfg_i: int  #0001
    cfg_f: int #0400

    max_moms_per_job: int
    momentum_list: list

    mom2_min: int 
    mom2_max : int

    run_eigs: bool
    eigs_slurm_nodes : int
    eigs_tasks_node : int 
    eigs_chroma_geometry : list
    eigs_chroma_minutes : int
    eigs_transfer_back : bool
    eigs_delete_after_transfer_back : bool

    run_perams: bool
    prop_slurm_nodes: int
    prop_tasks_nodes: int
    prop_chroma_geometry: list
    prop_chroma_minutes: int
    prop_num_gpu: int 

    run_meson:bool
    meson_slurm_nodes: int
    meson_tasks_nodes: int 
    meson_chroma_max_tslices_in_contraction: int
    meson_nvec: int
    meson_chroma_geometry: list
    meson_chroma_minutes: int
    meson_chroma_parts: int # split the time slices into this many different files

    # disco options 
    disco_slurm_nodes: int
    disco_chroma_geometry: list
    disco_chroma_minutes: int
    disco_num_gpu: int 
    
    #slurm job options 
    run_path: str 
    num_vecs:int
    num_vecs_perams:int

    job_name: str 
    ranks_per_node: int
    nodes: int 
    num_gpu: int 
    account: str 
    proj_name: str 
    facility: str 
    filename_out: str 
    job_name: str 
    superbblas_threads: int 
    omp_threads: int

    #paths to dirs
    code_dir: str 
    base_dir: str 
    colorvec_out: str 