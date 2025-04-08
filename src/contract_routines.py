from opt_einsum import contract
import numpy as np 
import src.gamma
from src.gamma import gamma
from src.ingest_data import load_elemental
gamma_i = [gamma[1], gamma[2], gamma[3], gamma[4]]


def contract_local(meson_file, nt, nvec, operator, t, mom):
    D0 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp')

    phi_0 = contract("ij,ab->ijab", operator.gamma, D0[0], optimize="optimal")
    phi_t = contract("ij,ab->ijab", operator.gamma, D0[t], optimize="optimal")
    return phi_0, phi_t


def contract_nabla(meson_file, nt, nvec, operator, t, mom):

    D1 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_1')
    D2 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_2')
    D3 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_3')

    nabla_0 = sum(
    contract("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[0] if i == 0 else D2[0] if i == 1 else D3[0], optimize="optimal") 
    for i in range(3)
        )
    
    nabla_t = sum(
        contract("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[t] if i == 0 else D2[t] if i == 1 else D3[t], optimize="optimal")
        for i in range(3)
    )
    return nabla_0, nabla_t

def contract_B_D(meson_file,nt,nvec,operator, t, mom,add=True):

    """
    Compute the gixBi and gixBi_t terms for B or D operators.
    The 'add' parameter determines whether to sum or subtract terms (B: subtract, D: add).
    """
    coeff = 1 if add else -1
    # load elementals displaced with two covariant derivatives
    D1D2 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_1_2')
    D2D1 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_2_1')

    D1D3 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_1_3')
    D3D1 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_3_1')

    D2D3 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_2_3')
    D3D2 = load_elemental(meson_file, nt, nvec, mom=mom, disp='disp_3_2')

    # src terms (t=0)
    D2D3_phi_0_1 = contract("ij,ab->ijab", gamma_i[0], D2D3[0], optimize="optimal")
    D2D3_phi_0_2 = contract("ij,ab->ijab", gamma_i[0], D3D2[0], optimize="optimal")  # Subtract this one

    D2D3_phi_0_3 = contract("ij,ab->ijab", gamma_i[1], D3D1[0], optimize="optimal")
    D2D3_phi_0_4 = contract("ij,ab->ijab", gamma_i[1], D1D3[0], optimize="optimal")  # Subtract this one

    D2D3_phi_0_5 = contract("ij,ab->ijab", gamma_i[2], D1D2[0], optimize="optimal")
    D2D3_phi_0_6 = contract("ij,ab->ijab", gamma_i[2], D2D1[0], optimize="optimal")  # Subtract this one

    gixBi = (D2D3_phi_0_1 - coeff * D2D3_phi_0_2 +
             D2D3_phi_0_3 - coeff * D2D3_phi_0_4 +
             D2D3_phi_0_5 - coeff * D2D3_phi_0_6)

    # snk terms (t)
    D2D3_phi_t_1 = contract("ij,ab->ijab", gamma_i[0], D2D3[t], optimize="optimal")
    D2D3_phi_t_2 = contract("ij,ab->ijab", gamma_i[0], D3D2[t], optimize="optimal")  # Subtract this one

    D2D3_phi_t_3 = contract("ij,ab->ijab", gamma_i[1], D3D1[t], optimize="optimal")
    D2D3_phi_t_4 = contract("ij,ab->ijab", gamma_i[1], D1D3[t], optimize="optimal")  # Subtract this one

    D2D3_phi_t_5 = contract("ij,ab->ijab", gamma_i[2], D1D2[t], optimize="optimal")
    D2D3_phi_t_6 = contract("ij,ab->ijab", gamma_i[2], D2D1[t], optimize="optimal")  # Subtract this one

    gixBi_t = (D2D3_phi_t_1 - coeff * D2D3_phi_t_2 +
               D2D3_phi_t_3 - coeff * D2D3_phi_t_4 +
               D2D3_phi_t_5 - coeff * D2D3_phi_t_6)

    return gixBi, gixBi_t
