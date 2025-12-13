#!/usr/bin/env python3
# correlator_factory.py — FINAL: full derivatives + vector support + per-op
from typing import Dict, Tuple
import h5py
import numpy as np
from opt_einsum import contract
import gamma
from file_io import DistillationObjectsIO
from dimeson_factory import BareOperator, DiMesonFactory


class CorrelatorFactory(DistillationObjectsIO):
    def __init__(self, ens: str, cfg_id: int, flavor_contents: list,
                 nvecs: int, lt: int, ntsrc: int, tsrc_step: int = 8,
                 data1: bool = False, collection: str | None = None):
        super().__init__(ens=ens, collection=collection)
        self.cfg_id = cfg_id
        self.flavor_contents = flavor_contents
        self.nvecs = nvecs
        self.lt = lt
        self.ntsrc = ntsrc
        self.tsrc_step = tsrc_step
        self.data1 = data1
        self._elemental_cache = {}

    # ------------------------------------------------------------------
    # Derivative displacement functions — FULLY RESTORED
    # ------------------------------------------------------------------
    def contract_local(self, operator, t: int, mom: str):
        D = self.get_elemental_block(mom, "disp")
        return contract("ij,ab->ijab", operator.base_gamma, D[t], optimize="optimal")

    def contract_nabla(self, operator, t: int, mom: str):
        gamma_i = [gamma.gamma[1], gamma.gamma[2], gamma.gamma[3]]
        D1 = self.get_elemental_block(mom, 'disp_1')
        D2 = self.get_elemental_block(mom, 'disp_2')
        D3 = self.get_elemental_block(mom, 'disp_3')

        def apply(t_slice):
            return sum(contract("ij,ab->ijab", operator.base_gamma @ gamma_i[i], 
                               D1[t_slice] if i == 0 else D2[t_slice] if i == 1 else D3[t_slice])
                       for i in range(3))

        return apply(0) if t == 0 else apply(t)

    def contract_B_D(self, operator, t: int, mom: str, add: bool = True):
        coeff = 1 if add else -1
        D = {
            "12": self.get_elemental_block(mom, "disp_1_2"),
            "21": self.get_elemental_block(mom, "disp_2_1"),
            "13": self.get_elemental_block(mom, "disp_1_3"),
            "31": self.get_elemental_block(mom, "disp_3_1"),
            "23": self.get_elemental_block(mom, "disp_2_3"),
            "32": self.get_elemental_block(mom, "disp_3_2"),
        }
        phi = (
            contract("ij,ab->ijab", gamma.gamma[1], D["23"][t]) -
            coeff * contract("ij,ab->ijab", gamma.gamma[1], D["32"][t]) +
            contract("ij,ab->ijab", gamma.gamma[2], D["31"][t]) -
            coeff * contract("ij,ab->ijab", gamma.gamma[2], D["13"][t]) +
            contract("ij,ab->ijab", gamma.gamma[3], D["12"][t]) -
            coeff * contract("ij,ab->ijab", gamma.gamma[3], D["21"][t])
        )
        return phi

    # ------------------------------------------------------------------
    # Correct gamma application — scalar + vector + derivatives
    # ------------------------------------------------------------------
    def _apply_gamma(self, op: BareOperator, D_block: np.ndarray) -> np.ndarray:
        """
        D_block: (4,4,64,64)
        Apply base gamma + optional polarization sum, exactly like contract_nabla
        """
        if op.gamma_i:
            # Vector/axial case: sum_i (gamma_i @ base_gamma) @ D_block
            result = sum(
                contract("ij,ab->ijab", gamma.gamma[i] @ op.base_gamma, D_block)
                for i in range(1, 4)
            )
            return result
        else:
            # Scalar/pseudoscalar: just base_gamma @ D_block
            return contract("ij,ab->ijab", op.base_gamma, D_block)

    def phi_sink(self, op: BareOperator, t: int, mom: str) -> np.ndarray:
        # Choose correct block based on derivative
        if op.derivative == "nabla":
            return self.contract_nabla(op, t, mom)
        elif op.derivative in {"B", "D"}:
            return self.contract_B_D(op, t, mom, add=(op.derivative == "B"))
        else:
            D = self.get_elemental_block(mom, "disp")
            return self._apply_gamma(op, D[t])

    def phi_source(self, op: BareOperator, mom: str) -> np.ndarray:
        if op.derivative == "nabla":
            return self.contract_nabla(op, 0, mom)
        elif op.derivative in {"B", "D"}:
            return self.contract_B_D(op, 0, mom, add=(op.derivative == "B"))
        else:
            D = self.get_elemental_block(mom, "disp")
            return self._apply_gamma(op, D[0])

    # ------------------------------------------------------------------
    # SINGLE DI-MESON OPERATOR — per-operator output
    # ------------------------------------------------------------------
    @classmethod
    def two_pt_single_dimeson(cls, proc, op1_src, op2_src, op1_snk, op2_snk,
                              h5_group, tsrc_avg=False):
        perams = proc.perambulators()
        LT, ntsrc = proc.lt, proc.ntsrc

        flavor_map = {
            "light_light": (perams["light_fwd"], perams["light_bwd"]),
            "light_charm": (perams["charm_fwd"], perams["light_bwd"]),
            "charm_light": (perams["light_fwd"], perams["charm_bwd"]),
            "charm_charm": (perams["charm_fwd"], perams["charm_bwd"]),
        }
        f1, f2 = proc.flavor_contents
        m1_fwd, m1_bwd = flavor_map[f1]
        m2_fwd, m2_bwd = flavor_map[f2]

        phi0_1 = proc.phi_source(op1_src, DiMesonFactory.mom_to_str(op1_src.mom))
        phi0_2 = proc.phi_source(op2_src, DiMesonFactory.mom_to_str(op2_src.mom))

        direct = np.zeros((ntsrc, LT), dtype=np.complex128)
        crossing = np.zeros((ntsrc, LT), dtype=np.complex128)
        meson1_corr = np.zeros((ntsrc, LT), dtype=np.complex128)
        meson2_corr = np.zeros((ntsrc, LT), dtype=np.complex128)

        for tsrc_idx in range(ntsrc):
            for t in range(LT):
                phi_t_1 = proc.phi_sink(op1_snk, t, DiMesonFactory.mom_to_str(op1_snk.mom))
                phi_t_2 = proc.phi_sink(op2_snk, t, DiMesonFactory.mom_to_str(op2_snk.mom))

                c1 = contract("ijab,jkbc,klcd,lida", phi_t_1, m1_fwd[tsrc_idx,t], phi0_1, m1_bwd[tsrc_idx,t])
                c2 = contract("ijab,jkbc,klcd,lida", phi_t_2, m2_fwd[tsrc_idx,t], phi0_2, m2_bwd[tsrc_idx,t])

                meson1_corr[tsrc_idx, t] = c1
                meson2_corr[tsrc_idx, t] = c2
                direct[tsrc_idx, t] = c1 * c2

                m1x = contract("ijab,jkbc,klcd,lida", phi_t_1, m1_fwd[tsrc_idx,t], phi0_1, m2_bwd[tsrc_idx,t])
                m2x = contract("ijab,jkbc,klcd,lida", phi_t_2, m2_fwd[tsrc_idx,t], phi0_2, m1_bwd[tsrc_idx,t])
                crossing[tsrc_idx, t] = m1x * m2x

        # Real part + tsrc avg
        direct = direct.real
        crossing = crossing.real
        meson1_corr = meson1_corr.real
        meson2_corr = meson2_corr.real

        if tsrc_avg:
            for i in range(ntsrc):
                sh = -proc.tsrc_step * i
                direct[i] = np.roll(direct[i], sh)
                crossing[i] = np.roll(crossing[i], sh)
                meson1_corr[i] = np.roll(meson1_corr[i], sh)
                meson2_corr[i] = np.roll(meson2_corr[i], sh)
            direct = direct.mean(axis=0)
            crossing = crossing.mean(axis=0)
            meson1_corr = meson1_corr.mean(axis=0)
            meson2_corr = meson2_corr.mean(axis=0)

        c15 = direct - crossing
        c6 = direct + crossing

        h5_group.create_dataset("dimeson_15", data=c15)
        h5_group.create_dataset("dimeson_6", data=c6)
        h5_group.create_dataset("meson1", data=meson1_corr)
        h5_group.create_dataset("meson2", data=meson2_corr)

        return True