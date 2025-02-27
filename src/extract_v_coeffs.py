import sys 
from dataclasses import dataclass 
from typing import Dict

@dataclass
class QuantumNum:
    had: int    # 1->meson (no parity in op name), 2->baryon (parity is in op name)
    flavor: str # Flavor
    N: int      # The N in SU(N)
    F: str      # Flavor irrep
    twoI: int   # 2*Isospin
    S: int      # Strangeness (s quark has strangeness -1)
    P: int      # Rest-frame parity
    G: int      # G-parity (if it exists)

# Initialize classnames
Class_t = Dict[str, QuantumNum]

classnames: Class_t = {
    "pion": QuantumNum(had=1, flavor="pion", N=2, F="3", twoI=2, S=0, P=-1, G=-1),
}

def get_flavor_index(op_name: str) -> QuantumNum:
    """Get flavor of particle id"""
    for k, v in classnames.items():
        if op_name.startswith(k):
            return v
    raise ValueError(f"ERROR: cannot extract flavor from operator name {op_name}")

# ----------------------------------------------------------------------------------
def get_sunf(op_name: str) -> int:
    """Get the N in SU(N)"""
    return get_flavor_index(op_name).N

# ----------------------------------------------------------------------------------
def get_flavor_irrep(op_name: str) -> str:
    """Get flavor irrep"""
    return get_flavor_index(op_name).F

# ----------------------------------------------------------------------------------
def get_flavor(op_name: str) -> str:
    """Get flavor of particle id"""
    return get_flavor_index(op_name).flavor

# ----------------------------------------------------------------------------------
def get_num_quarks(op_name: str) -> int:
    """Get number of quarks"""
    had = get_flavor_index(op_name).had
    if had == 1:
        return 2  # meson
    elif had == 2:
        return 3  # baryon
    elif had == 3:
        return 2
    else:
        raise ValueError(f"Unknown hadron type: {had}")
    
# ----------------------------------------------------------------------------------
def get_two_isospin(op_name: str) -> int:
    """Get 2*Isospin"""
    return get_flavor_index(op_name).twoI

# ----------------------------------------------------------------------------------
def get_three_y(op_name: str) -> int:
    """Get 3*Y of particle id"""
    qq = get_flavor_index(op_name)
    
    # Y = S + B
    three_y = qq.S
    if qq.had == 2:
        three_y += 1  # baryon

    return 3 * three_y

# ----------------------------------------------------------------------------------
def get_g_parity(op_name: str) -> int:
    """Get G-parity of particle id"""
    return get_flavor_index(op_name).G

# ----------------------------------------------------------------------------------
def get_strangeness(op_name: str) -> int:
    """Get strangeness of particle id"""
    return get_flavor_index(op_name).S

# ----------------------------------------------------------------------------------
def get_irrep_no_parity(op_name: str) -> str:
    """Get irrep without parity"""
    return get_cubic_rep_no_parity(get_irrep(op_name))

# ----------------------------------------------------------------------------------
def get_irrep_with_parity(op_name: str) -> str:
    """Get irrep with parity"""
    irrep = get_cubic_rep_no_parity(get_irrep(op_name))
    mm = get_flavor_index(op_name)
    
    return build_irrep_with_parity(irrep, mm.P)

# ----------------------------------------------------------------------------------
def get_irrep_with_g_parity(op_name: str) -> str:
    """Get irrep with G-parity"""
    irrep = get_cubic_rep_no_parity(get_irrep(op_name))
    mm = get_flavor_index(op_name)
    
    return build_irrep_with_parity(irrep, mm.P)

# ----------------------------------------------------------------------------------
def get_irrep_with_pg(op_name: str) -> str:
    """Get irrep with parity and G-parity"""
    irrep = get_cubic_rep_no_parity(get_irrep(op_name))
    mm = get_flavor_index(op_name)
    
    return build_irrep_with_pg(irrep, mm.P, mm.G)



def flip_charge_conjugation():

    # Process each line from standard input
    for line in sys.stdin:
        line = line.strip()
        parts = line.split()

        opname = parts[0]
        subopname = parts[1]
        val = float(parts[2])

        C = 1  # Initialize C as +1

        # Gamma matrices
        if any(x in subopname for x in ["_a0x", "_a1x", "_pionx", "_pion_2x"]):
            C *= 1
        elif any(x in subopname for x in ["_b0x", "_b1x", "_rhox", "_rho_2x"]):
            C *= -1
        else:
            raise ValueError(f"Unknown gamma matrix structure for op = {subopname}")

        # Derivatives
        if any(x in subopname for x in ["xD0_", "xD2_J0", "xD2_J2", "xD3_J131"]):
            C *= 1
        elif any(x in subopname for x in ["xD1", "xD2_J1", "xD3_J130", "xD3_J132"]):
            C *= -1
        else:
            raise ValueError(f"Unknown derivative structure for op = {subopname}")

        # Output the modified values
        print(f"{opname} {subopname} {C * val:.6g}")

def charge_conjugation(operator_map):

    for k,v in operator_map:
