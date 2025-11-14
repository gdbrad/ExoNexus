from typing import List, Dict, Tuple, Any
import h5py
import numpy as np
from insertion_factory import gamma
import file_io
from file_io import DistillationObjectsIO
#from operator_factory import MHI, MHICollection
from dimeson_factory import DiMesonOperator,BareOperator
import re
import os 
from ingest_data import load_elemental, reverse_perambulator_time
from itertools import product
from opt_einsum import contract # this can be replaced with np.einsum

# ----------------------------------------------------------------------
# Processor – correlator logic
# ----------------------------------------------------------------------
class CorrelatorFactory(DistillationObjectsIO):
    def __init__(
        self,
        ens: str,
        cfg_id: int,
        flavor_contents: List[str],
        nvecs: int,
        lt: int,
        ntsrc: int,
        tsrc_step: int = 8,
        data1: bool = False,
        collection: str | None = None,
    ) -> None:
        super().__init__(ens=ens, collection=collection)
        self.cfg_id = cfg_id
        self.flavor_contents = flavor_contents
        self.nvecs = nvecs
        self.lt = lt
        self.ntsrc = ntsrc
        self.tsrc_step = tsrc_step
        self.data1 = data1
        #self.load_distillation_objects()
        # cache for elementals (mom, disp) → array
        self._elemental_cache: Dict[Tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # derivative displacement functions integrated as methods which use the cache
    # ------------------------------------------------------------------
    # import os 
    # def _meson_path(self) -> str:
    #     return os.path.join(self.dirs["meson"], f"meson-{self.nvecs}_cfg{self.cfg_id}.h5")
    
       # ------------------------------------------------------------------
    # Use get_elemental_block from parent (on-demand from full file)
    # ------------------------------------------------------------------
    def contract_local(self, operator, t: int, mom: str):
        D = self.get_elemental_block(mom, "disp")
        phi = contract("ij,ab->ijab", operator.gamma, D[t], optimize="optimal")
        return phi

    def contract_nabla(self, operator, t: int, mom: str):
        gamma_i = [gamma.gamma[1], gamma.gamma[2], gamma.gamma[3]]
        D1 = self.get_elemental_block(mom, 'disp_1')
        D2 = self.get_elemental_block(mom, 'disp_2')
        D3 = self.get_elemental_block(mom, 'disp_3')

        nabla_0 = sum(
        np.einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[0] if i == 0 else D2[0] if i == 1 else D3[0], optimize="optimal") 
        for i in range(3)
            )
    
        nabla_t = sum(
            np.einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[t] if i == 0 else D2[t] if i == 1 else D3[t], optimize="optimal")
            for i in range(3)
        )
        if t==0:
            return nabla_0
        else:
            return nabla_t

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
            contract("ij,ab->ijab", gamma.gamma[1], D["23"][t], optimize="optimal") -
            coeff * contract("ij,ab->ijab", gamma.gamma[1], D["32"][t], optimize="optimal") +
            contract("ij,ab->ijab", gamma.gamma[2], D["31"][t], optimize="optimal") -
            coeff * contract("ij,ab->ijab", gamma.gamma[2], D["13"][t], optimize="optimal") +
            contract("ij,ab->ijab", gamma.gamma[3], D["12"][t], optimize="optimal") -
            coeff * contract("ij,ab->ijab", gamma.gamma[3], D["21"][t], optimize="optimal")
        )
        return phi
    
        # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    @classmethod
    def two_pt(
        cls,
        proc: "CorrelatorFactory",
        h5_group: h5py.Group,
        tsrc_avg: bool = False,
        three_bar: bool = False,
    ) -> bool:
        """
        Entry point used by `two_pt_corr.py`.
        1. Load *all* required perambulators (light, charm, …)
        2. Build the forward/backward map
        3. Compute the di-meson matrix
        """
        # --------------------------------------------------------------
        # 1. Load everything that belongs to the system
        # --------------------------------------------------------------
        # `proc` is a CorrelatorFactory that already knows cfg_id,
        # flavor_contents, nvecs, lt, … – we just have to ask it to load.
        try:
            proc.load_for_system("Dpi")          # <-- **THIS IS THE MISSING CALL**
        except Exception as e:
            print(f"[ERROR] Failed to load perambulators for cfg {proc.cfg_id}: {e}")
            return False

        # --------------------------------------------------------------
        # 2. Build forward / backward perambulator map
        # --------------------------------------------------------------
        try:
            peram_data = proc._peram_data()      # <-- now safe (perams are loaded)
        except Exception as e:
            print(f"[ERROR] _peram_data() failed for cfg {proc.cfg_id}: {e}")
            return False

        # --------------------------------------------------------------
        # 3. Generate operators and compute the matrix
        # --------------------------------------------------------------
        if len(proc.flavor_contents) != 2:
            print(f"[WARN] Expected exactly 2 flavor contents, got {proc.flavor_contents}")
            return False

        operators = DiMesonOperator.generate_operators()
        cls.di_meson_correlator_matrix(
            operators=operators,
            proc=proc,
            h5_group=h5_group,
            peram_data=peram_data,
            tsrc_avg=tsrc_avg,
            three_bar=three_bar,
        )
        return True
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 1. Di-meson case
        # ------------------------------------------------------------------
        # if len(proc.flavor_contents) == 2:
        #     proc.di_meson_correlator(
        #         h5_group=h5_group,
        #         peram_data=peram_data,
        #         phi_0=phi_0,
        #         tsrc_avg=tsrc_avg,
        #         three_bar=three_bar,
        #     )
        # ------------------------------------------------------------------
        # 3. Single-meson case
        # ------------------------------------------------------------------
        # else:
        #     proc.single_meson_correlator(
        #         h5_group=h5_group,
        #         peram_data=peram_data,
        #         phi_0=phi_0,
        #         tsrc_avg=tsrc_avg,
        #     )
        # # ------------------------------------------------------------------
        # # 3. Always write the *individual* meson correlators for every flavour
        # # ------------------------------------------------------------------
        # for idx, fc in enumerate(proc.flavor_contents, 1):
        #     peram, peram_b = peram_data[fc]
        #     single = np.zeros((num_tsrc, LT), dtype=np.cdouble)
        #     for tsrc_idx in range(num_tsrc):
        #         for dt in range(LT):
        #             phi_t = contract(
        #                 "ij,ab->ijab",
        #                 gamma.gamma[5],
        #                 proc.meson_elemental[dt],
        #                 optimize="optimal",
        #             )
        #             tau = peram[tsrc_idx, dt, :, :, :, :]
        #             tau_b = peram_b[tsrc_idx, dt, :, :, :, :]
        #             single[tsrc_idx, dt] = contract(
        #                 "ijab,jkbc,klcd,lida",
        #                 phi_t,
        #                 tau,
        #                 phi_0,
        #                 tau_b,
        #                 optimize="optimal",
        #             )
        #     single = single.real
        #     if tsrc_avg:
        #         for i in range(num_tsrc):
        #             single[i] = np.roll(single[i], -tsrc_step * i)
        #         single = single.mean(axis=0)
        #     key = f"meson{idx}_{fc}"
        #     suffix = "_tsrc_avg" if tsrc_avg else ""
        #     h5_group.create_dataset(f"{key}/cfg_{proc.cfg_id}{suffix}", data=single)
        # print(f"Cfg {proc.cfg_id} processed – results written.")

    # ------------------------------------------------------------------
    # Single-meson correlator
    # ------------------------------------------------------------------
    @classmethod
    def single_meson_correlator(
        cls,
        proc: "CorrelatorFactory",
        h5_group: h5py.Group,
        peram_data: Dict[str, Tuple[Any, Any]],
        phi_0: np.ndarray,
        tsrc_avg: bool,
    ) -> None:
        fc = proc.flavor_contents[0]
        peram, peram_b = peram_data[fc]
        LT = proc.lt
        num_tsrc = proc.ntsrc
        tsrc_step = proc.tsrc_step
        data = np.zeros((num_tsrc, LT), dtype=np.cdouble)
        for tsrc_idx in range(num_tsrc):
            for dt in range(LT):
                phi_t = contract(
                    "ij,ab->ijab",
                    gamma.gamma[5],
                    proc.meson_elemental[dt],
                    optimize="optimal",
                )
                tau = peram[tsrc_idx, dt, :, :, :, :]
                tau_b = peram_b[tsrc_idx, dt, :, :, :, :]
                data[tsrc_idx, dt] = contract(
                    "ijab,jkbc,klcd,lida",
                    phi_t,
                    tau,
                    phi_0,
                    tau_b,
                    optimize="optimal",
                )
        data = data.real
        if tsrc_avg:
            for i in range(num_tsrc):
                data[i] = np.roll(data[i], -tsrc_step * i)
            data = data.mean(axis=0)
        prefix = f"meson1_{fc}"
        suffix = "_tsrc_avg" if tsrc_avg else ""
        h5_group.create_dataset(f"{prefix}/cfg_{proc.cfg_id}{suffix}", data=data)

    # ------------------------------------------------------------------
    # Di-meson correlator (single operator)
    #
    # Direct:
    # D A (creation operator): <phi_t_A * tau_A * phi_0_A * tau_A_back>
    # D B (creation operator): <phi_t_B * tau_B * phi_0_B * tau_B_back>
    # pi-pi creation operator = pion A x pion B 
    # Di-meson direct: C_A(t) * C_B(t), where C_A and C_B are the individual meson correlators.
    # ------------------------------------------------------------------
    @classmethod
    def di_meson_correlator(
        cls,
        proc: "CorrelatorFactory",
        h5_group: h5py.Group,
        peram_data: Dict[str, Tuple[Any, Any]],
        phi_0: np.ndarray,
        tsrc_avg: bool,
        three_bar: bool,
    ) -> None:
        f1, f2 = proc.flavor_contents
        group_name = proc.get_meson_system_name()
        identical = f1 == f2
        LT = proc.lt
        num_tsrc = proc.ntsrc
        tsrc_step = proc.tsrc_step
        direct = np.zeros((num_tsrc, LT), dtype=np.cdouble)
        crossing = np.zeros((num_tsrc, LT), dtype=np.cdouble) if not identical else None
        disc = np.zeros((num_tsrc, LT), dtype=np.cdouble) if three_bar else None
        p1, p1b = peram_data[f1]
        p2, p2b = peram_data[f2]
        for tsrc_idx in range(num_tsrc):
            for t in range(LT):
                phi_t = contract(
                    "ij,ab->ijab",
                    gamma.gamma[5],
                    proc.meson_elemental[t],
                    optimize="optimal",
                )
                # direct legs
                tau1 = p1[tsrc_idx, t, :, :, :, :]
                tau1b = p1b[tsrc_idx, t, :, :, :, :]
                m1 = contract(
                    "ijab,jkbc,klcd,lida",
                    phi_t,
                    tau1,
                    phi_0,
                    tau1b,
                    optimize="optimal",
                )
                tau2 = p2[tsrc_idx, t, :, :, :, :]
                tau2b = p2b[tsrc_idx, t, :, :, :, :]
                m2 = contract(
                    "ijab,jkbc,klcd,lida",
                    phi_t,
                    tau2,
                    phi_0,
                    tau2b,
                    optimize="optimal",
                )
                direct[tsrc_idx, t] = m1 * m2
                # crossing
                if not identical:
                    m1c = contract(
                        "ijab,jkbc,klcd,lida",
                        phi_t,
                        tau1,
                        phi_0,
                        p2b[tsrc_idx, t, :, :, :, :],
                        optimize="optimal",
                    )
                    m2c = contract(
                        "ijab,jkbc,klcd,lida",
                        phi_t,
                        tau2,
                        phi_0,
                        p1b[tsrc_idx, t, :, :, :, :],
                        optimize="optimal",
                    )
                    crossing[tsrc_idx, t] = m1c * m2c
                # disconnected loop for bar{3}
                if three_bar:
                    loop = contract(
                        "siab,sjcd->abcd",
                        p2[tsrc_idx, t, :, :, :, :],
                        p2b[tsrc_idx, t, :, :, :, :],
                        optimize="optimal",
                    )
                    disc[tsrc_idx, t] = m1 * contract(
                        "ijab,abcd->", phi_t, loop, optimize="optimal"
                    )
        # real part
        direct = direct.real
        if not identical:
            crossing = crossing.real
        if three_bar:
            disc = disc.real
        # t_src averaging
        if tsrc_avg:
            for i in range(num_tsrc):
                shift = -tsrc_step * i
                direct[i] = np.roll(direct[i], shift)
                if not identical:
                    crossing[i] = np.roll(crossing[i], shift)
                if three_bar:
                    disc[i] = np.roll(disc[i], shift)
            direct = direct.mean(axis=0)
            crossing = crossing.mean(axis=0) if not identical else None
            disc = disc.mean(axis=0) if three_bar else None
        # irreps
        if identical:
            corr15 = direct
            corr6 = direct
            corr3b = direct - (8.0 / 3.0) * disc if three_bar else None
        else:
            corr15 = direct - crossing
            corr6 = direct + crossing
            corr3b = (
                direct + (1.0 / 3.0) * crossing - (8.0 / 3.0) * disc
                if three_bar
                else None
            )
        # write
        base = f"{group_name}"
        suffix = "_tsrc_avg" if tsrc_avg else ""
        h5_group.create_dataset(f"{base}/direct/cfg_{proc.cfg_id}{suffix}", data=direct)
        if not identical:
            h5_group.create_dataset(f"{base}/crossing/cfg_{proc.cfg_id}{suffix}", data=crossing)
        h5_group.create_dataset(f"{base}/15/cfg_{proc.cfg_id}{suffix}", data=corr15)
        h5_group.create_dataset(f"{base}/6/cfg_{proc.cfg_id}{suffix}", data=corr6)
        if three_bar:
            h5_group.create_dataset(f"{base}/disconnected/cfg_{proc.cfg_id}{suffix}", data=disc)
            h5_group.create_dataset(f"{base}/3_bar/cfg_{proc.cfg_id}{suffix}", data=corr3b)

    # ----------------------------------------------------------------------------
    # Di-meson correlator matrix for a given irrep, total momenta P^2 (>1 operators)
    # -----------------------------------------------------------------------------
    @classmethod
    def di_meson_correlator_matrix(
        cls,
        operators: Dict[str, DiMesonOperator],
        proc: "CorrelatorFactory",
        h5_group: h5py.Group,           # ← not used any more (kept for signature)
        peram_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        tsrc_avg: bool,
        three_bar: bool,
    ) -> None:

        f1, f2 = proc.flavor_contents
        identical = f1 == f2
        LT = proc.lt
        num_tsrc = proc.ntsrc
        num_op = len(operators)

        print(f"\n[PER-PAIR OUTPUT] Writing one file per operator pair → {num_op**2} files total")

        p1, p1b = peram_data[f1]
        p2, p2b = peram_data[f2]

        def contract_op(op: BareOperator, t: int, mom: str):
            if op.deriv is None:
                return proc.contract_local(op, t, mom)
            if op.deriv == "nabla":
                return proc.contract_nabla(op, t, mom)
            add = op.deriv == "D"
            return proc.contract_B_D(op, t, mom, add=add)

        # ------------------------------------------------------------------
        # Main loop – one file per (src, snk) pair
        # ------------------------------------------------------------------
        total_pairs = num_op * num_op
        done = 0

        for src_idx, (src_name, src_op) in enumerate(operators.items()):
            phi_0_D  = contract_op(src_op.op1, 0, src_op.op1.mom)
            phi_0_pi = contract_op(src_op.op2, 0, src_op.op2.mom)

            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                done += 1
                pair_label = f"{src_op.name}_X_{snk_op.name}"
                outfile = f"Dpi_cfg{proc.cfg_id:04d}_{pair_label}.h5"

                print(f"[{done:03d}/{total_pairs}] → {outfile}")

                # Temporary arrays for this pair only
                direct   = np.zeros((num_tsrc, LT), dtype=np.cdouble)
                crossing = np.zeros((num_tsrc, LT), dtype=np.cdouble) if not identical else None
                disc     = np.zeros((num_tsrc, LT), dtype=np.cdouble) if three_bar else None

                # ---- Contraction for this single pair ----
                for tsrc_idx in range(num_tsrc):
                    for t in range(LT):
                        phi_t_D  = contract_op(snk_op.op1, t, snk_op.op1.mom)
                        phi_t_pi = contract_op(snk_op.op2, t, snk_op.op2.mom)

                        # Direct
                        m1 = contract("ijab,jkbc,klcd,lida",
                                      phi_t_D,  p1[tsrc_idx, t, :, :, :, :],
                                      phi_0_D,  p1b[tsrc_idx, t, :, :, :, :], optimize="optimal")
                        m2 = contract("ijab,jkbc,klcd,lida",
                                      phi_t_pi, p2[tsrc_idx, t, :, :, :, :],
                                      phi_0_pi, p2b[tsrc_idx, t, :, :, :, :], optimize="optimal")
                        direct[tsrc_idx, t] = m1 * m2

                        if not identical:
                            m1c = contract("ijab,jkbc,klcd,lida",
                                           phi_t_D, p1[tsrc_idx, t, :, :, :, :],
                                           phi_0_D, p2b[tsrc_idx, t, :, :, :, :], optimize="optimal")
                            m2c = contract("ijab,jkbc,klcd,lida",
                                           phi_t_pi, p2[tsrc_idx, t, :, :, :, :],
                                           phi_0_pi, p1b[tsrc_idx, t, :, :, :, :], optimize="optimal")
                            crossing[tsrc_idx, t] = m1c * m2c

                        if three_bar:
                            loop = contract("siab,sjcd->abcd",
                                            p2[tsrc_idx, t, :, :, :, :],
                                            p2b[tsrc_idx, t, :, :, :, :], optimize="optimal")
                            disc[tsrc_idx, t] = m1 * contract("ijab,abcd->", phi_t_D, loop, optimize="optimal")

                # ---- Post-processing ----
                direct = direct.real
                if crossing is not None:
                    crossing = crossing.real
                if disc is not None:
                    disc = disc.real

                if tsrc_avg:
                    for i in range(num_tsrc):
                        shift = -proc.tsrc_step * i
                        direct[i] = np.roll(direct[i], shift, axis=-1)
                        if crossing is not None:
                            crossing[i] = np.roll(crossing[i], shift, axis=-1)
                        if disc is not None:
                            disc[i] = np.roll(disc[i], shift, axis=-1)
                    direct = direct.mean(axis=0)
                    crossing = crossing.mean(axis=0) if crossing is not None else None
                    disc = disc.mean(axis=0) if disc is not None else None

                # Final irreps
                if identical:
                    c15 = direct
                    c6  = direct
                    c3b = direct - (8.0/3.0)*disc if three_bar else None
                else:
                    c15 = direct - crossing
                    c6  = direct + crossing
                    c3b = direct + (1.0/3.0)*crossing - (8.0/3.0)*disc if three_bar else None

                # ---- Write one clean file ----
                suffix = "_tsrc_avg" if tsrc_avg else ""
                with h5py.File(outfile, "w", libver='latest') as f:
                    f.create_dataset("direct",      data=direct)
                    if not identical:
                        f.create_dataset("crossing", data=crossing)
                    f.create_dataset("15",          data=c15)
                    f.create_dataset("6",           data=c6)
                    if three_bar:
                        f.create_dataset("disconnected", data=disc)
                        f.create_dataset("3_bar",        data=c3b)

                    # Metadata
                    f.attrs["source_operator"] = src_op.name
                    f.attrs["sink_operator"]   = snk_op.name
                    f.attrs["cfg"]             = proc.cfg_id
                    f.attrs["tsrc_avg"]        = tsrc_avg
                    f.attrs["three_bar"]       = three_bar

                print(f"    → written {os.path.getsize(outfile)/1024:.1f} kB")

        print(f"\nALL DONE! {total_pairs} clean HDF5 files written.")
        print(f"   Directory now contains files like:")
        print(f"   Dpi_cfg{proc.cfg_id:04d}_D_000_rho_nabla_a1p_X_pi_000_rho_nabla_a1p.h5")