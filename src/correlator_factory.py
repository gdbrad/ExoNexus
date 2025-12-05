from typing import List, Dict, Tuple, Any
import h5py
import numpy as np
from itertools import product
from opt_einsum import contract, contract_path # this can be replaced with np.einsum
import functools
import tqdm 

import gamma
from file_io import DistillationObjectsIO
from dimeson_factory import DiMesonOperator,BareOperator

# ----------------------------------------------------------------------
# Processor – correlator logic
# ----------------------------------------------------------------------
class CorrelatorFactory(DistillationObjectsIO):
    def __init__(
        self,
        ens: str,
        cfg_id: int,
        flavor_contents: List[str],
        collection: str | None = None,
    ) -> None:
        super().__init__(ens=ens, collection=collection)
        self.cfg_id = cfg_id
        self.flavor_contents = flavor_contents
        #self.load_distillation_objects()
        # cache for elementals (mom, disp) → array
        self._elemental_cache: Dict[Tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # derivative displacement functions integrated as methods which use the cache
    # ------------------------------------------------------------------
    def contract_local(self, operator, t: int, mom: str):
        D = self.get_elemental_block(mom, "disp")
        return contract("ij,ab->ijab", operator.gamma, D[t], optimize="optimal")  # ← +γ₅ always
    
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
            proc.load_for_system("Dpi")       
        except Exception as e:
            print(f"[ERROR] Failed to load perambulators for cfg {proc.cfg_id}: {e}")
            return False

        # --------------------------------------------------------------
        # 2. Build forward / backward perambulator map
        # --------------------------------------------------------------
        try:
            #peram_data = proc._peram_data()  
            peram_data = proc.perambulators()  

        except Exception as e:
            print(f"[ERROR] _peram_data() failed for cfg {proc.cfg_id}: {e}")
            return False

        # --------------------------------------------------------------
        # 3. Generate operators and compute the matrix
        # --------------------------------------------------------------
        if len(proc.flavor_contents) != 2:
            print(f"[WARN] Expected exactly 2 flavor contents, got {proc.flavor_contents}")
            return False

        operators = {op.name: op for op in DiMesonOperator.get_ordered()}
        cls.di_meson_correlator_matrix(
            operators=operators,
            proc=proc,
            h5_group=h5_group,
            #peram_data=peram_data,
            tsrc_avg=tsrc_avg,
            three_bar=three_bar,
        )
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
        # # # ------------------------------------------------------------------
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

        return True

    # ------------------------------------------------------------------
    # Single-meson correlator
    # ------------------------------------------------------------------
    # @classmethod
    # def single_meson_correlator(
    #     cls,
    #     proc: "CorrelatorFactory",
    #     h5_group: h5py.Group,
    #     peram_data:  Dict[str, Tuple[np.ndarray, np.ndarray]],
    #     tsrc_avg: bool,
    # ) -> None:
    #     fc = proc.flavor_contents[0]
    #     peram, peram_b = peram_data[fc]
    #     LT = proc.lt
    #     num_tsrc = proc.ntsrc
    #     tsrc_step = proc.tsrc_step
    #     data = np.zeros((num_tsrc, LT), dtype=np.cdouble)
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
    #             data[tsrc_idx, dt] = contract(
    #                 "ijab,jkbc,klcd,lida",
    #                 phi_t,
    #                 tau,
    #                 phi_0,
    #                 tau_b,
    #                 optimize="optimal",
    #             )
    #     data = data.real
    #     if tsrc_avg:
    #         for i in range(num_tsrc):
    #             data[i] = np.roll(data[i], -tsrc_step * i)
    #         data = data.mean(axis=0)
    #     prefix = f"meson1_{fc}"
    #     suffix = "_tsrc_avg" if tsrc_avg else ""
    #     h5_group.create_dataset(f"{prefix}/cfg_{proc.cfg_id}{suffix}", data=data)

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

    # ----------------------------------------------------------------------------
# Di-meson correlator matrix 
# ----------------------------------------------------------------------------

    # @classmethod
    # def di_meson_correlator_matrix(
    #     cls,
    #     operators: Dict[str, DiMesonOperator],
    #     proc: "CorrelatorFactory",
    #     h5_group: h5py.Group,
    #     tsrc_avg: bool,
    #     three_bar: bool,
    # ) -> None:
    #     import time
    #     start = time.time()

    #     # ------------------------------------------------------------------
    #     # 1. Get perambulators via the old, clean _peram_data() method
    #     # ------------------------------------------------------------------
    #     peram_data = proc._peram_data()   # ← back to your trusted way

    #     LT = proc.lt
    #     ntsrc = proc.ntsrc
    #     tsrc_step = proc.tsrc_step
    #     op_list = list(operators.values())
    #     num_op = len(op_list)

    #     print(f"[MATRIX] Computing {num_op}×{num_op} Dπ matrix, ntsrc={ntsrc}, Lt={LT}")

    #     # ------------------------------------------------------------------
    #     # 2. Extract forward/backward legs for the two mesons
    #     # ------------------------------------------------------------------
    #     f1, f2 = proc.flavor_contents
    #     meson1_fwd, meson1_bwd = peram_data[f1]   # e.g. charm_light → (light_fwd, charm_bwd)
    #     meson2_fwd, meson2_bwd = peram_data[f2]   # e.g. light_light → (light_fwd, light_bwd)

    #     # ------------------------------------------------------------------
    #     # 3. Interpolator builders — no signs here!
    #     # ------------------------------------------------------------------
    #     def phi_sink(op: BareOperator, t:int, mom: str):
    #         D = proc.get_elemental_block(mom, "disp")
    #         return contract("ij,ab->ijab", op.gamma, D[t], optimize="optimal")

    #     def phi_source(op: BareOperator, mom: str):
    #         D = proc.get_elemental_block(mom, "disp")
    #         return contract("ij,ab->ijab", op.gamma, D[0], optimize="optimal")

    #     # Pre-cache source interpolators
    #     phi0_m1_cache = [phi_source(op.op1, op.op1.mom) for op in op_list]  # D source for meson1 (D)
    #     phi0_m2_cache = [phi_source(op.op2, op.op2.mom) for op in op_list]  # source for meson2 (π)

    #     # ------------------------------------------------------------------
    #     # 4. Single-meson correlators
    #     # ------------------------------------------------------------------
    #     first_op = op_list[0]
    #     phi0_m1_first = phi_source(first_op.op1, first_op.op1.mom)
    #     phi0_m2_first = phi_source(first_op.op2, first_op.op2.mom)

    #     meson1_corr = np.zeros((ntsrc, LT), dtype=np.cdouble)
    #     meson2_corr = np.zeros((ntsrc, LT), dtype=np.cdouble)

    #     for tsrc_idx in range(ntsrc):
    #         for t in range(LT):
    #             phi_t_1 = phi_sink(first_op.op1, t, first_op.op1.mom)
    #             phi_t_2 = phi_sink(first_op.op2, t, first_op.op2.mom)

    #             c1 = contract("ijab,jkbc,klcd,lida",
    #                         phi_t_1, meson1_fwd[tsrc_idx, t], phi0_m1_first, meson1_bwd[tsrc_idx, t],
    #                         optimize="optimal")
    #             c2 = contract("ijab,jkbc,klcd,lida",
    #                         phi_t_2, meson2_fwd[tsrc_idx, t], phi0_m2_first, meson2_bwd[tsrc_idx, t],
    #                         optimize="optimal")

    #             meson1_corr[tsrc_idx, t] = c1
    #             meson2_corr[tsrc_idx, t] = c2

    #     h5_group.create_dataset("meson1_correlator", data=meson1_corr.real)
    #     h5_group.create_dataset("meson2_correlator", data=meson2_corr.real)
    #     print("   → meson1 (D) and meson2 (π) correlators written")

    #     # ------------------------------------------------------------------
    #     # 5. Main di-meson matrix loop
    #     # ------------------------------------------------------------------
    #     for src_idx, src_op in enumerate(op_list):
    #         phi0_m1_src = phi0_m1_cache[src_idx]
    #         phi0_m2_src = phi0_m2_cache[src_idx]

    #         for snk_idx, snk_op in enumerate(op_list):
    #             direct   = np.zeros((ntsrc, LT), dtype=np.cdouble)
    #             crossing = np.zeros((ntsrc, LT), dtype=np.cdouble)

    #             for tsrc_idx in range(ntsrc):
    #                 for t in range(LT):
    #                     phi_t_1 = phi_sink(snk_op.op1, t, snk_op.op1.mom)
    #                     phi_t_2 = phi_sink(snk_op.op2, t, snk_op.op2.mom)

    #                     # Direct
    #                     m1 = contract("ijab,jkbc,klcd,lida",
    #                                 phi_t_1, meson1_fwd[tsrc_idx,t], phi0_m1_src,
    #                                 meson1_bwd[tsrc_idx,t], optimize="optimal")
    #                     m2 = contract("ijab,jkbc,klcd,lida",
    #                                 phi_t_2, meson2_fwd[tsrc_idx,t], phi0_m2_src,
    #                                 meson2_bwd[tsrc_idx,t], optimize="optimal")
    #                     direct[tsrc_idx, t] = m1 * m2

    #                     # Crossing
    #                     m1x = contract("ijab,jkbc,klcd,lida",
    #                                 phi_t_1, meson1_fwd[tsrc_idx,t], phi0_m1_src,
    #                                 meson2_bwd[tsrc_idx,t], optimize="optimal")
    #                     m2x = contract("ijab,jkbc,klcd,lida",
    #                                 phi_t_2, meson2_fwd[tsrc_idx,t], phi0_m2_src,
    #                                 meson1_bwd[tsrc_idx,t], optimize="optimal")
    #                     crossing[tsrc_idx, t] = m1x * m2x

    #             # Post-processing
    #             direct   = direct.real
    #             crossing = crossing.real

    #             if tsrc_avg:
    #                 for i in range(ntsrc):
    #                     sh = -tsrc_step * i
    #                     direct[i]   = np.roll(direct[i], sh)
    #                     crossing[i] = np.roll(crossing[i], sh)
    #                 direct   = direct.mean(axis=0)
    #                 crossing = crossing.mean(axis=0)

    #             c15 = direct - crossing
    #             c6  = direct + crossing

    #             grp = h5_group.create_group(f"op{src_idx:02d}_op{snk_idx:02d}")
    #             grp.attrs["src"] = src_op.name
    #             grp.attrs["snk"] = snk_op.name
    #             grp.create_dataset("15", data=c15)
    #             grp.create_dataset("6",  data=c6)

    #     print(f"[DONE] Full matrix written in {(time.time()-start)/60:.1f} min")
    @classmethod
    def di_meson_correlator_matrix(
        cls,
        operators: Dict[str, DiMesonOperator],
        proc: "CorrelatorFactory",
        h5_group: h5py.Group,
        tsrc_avg: bool,
        three_bar: bool,
    ) -> None:
        import time
        start = time.time()

        # ------------------------------------------------------------------
        # 1. Load explicit, correctly signed perambulators
        # ------------------------------------------------------------------
        perams = proc.perambulators()  # ← light_fwd, light_bwd, charm_fwd, charm_bwd
        ntsrc = proc.ntsrc
        LT = proc.lt
        tsrc_step = proc.tsrc_step
        op_list = list(operators.values())
        num_op = len(op_list)

        print(f"[MATRIX] Computing {num_op}×{num_op} Dπ matrix, ntsrc={ntsrc}, Lt={LT}")

        # ------------------------------------------------------------------
        # 2. Map flavor strings → correct (forward, backward) perambulators
        # ------------------------------------------------------------------
        flavor_map = {
            "light_light":  (perams["light_fwd"],  perams["light_bwd"]),
            "light_charm":  (perams["charm_fwd"],  perams["light_bwd"]),   # rare: D⁻
            "charm_light":  (perams["light_fwd"],  perams["charm_bwd"]),   # standard D⁺/D⁰
            "charm_charm":  (perams["charm_fwd"],  perams["charm_bwd"]),
            # add "strange_light", "light_strange", etc. later
        }

        f1, f2 = proc.flavor_contents
        if f1 not in flavor_map or f2 not in flavor_map:
            raise ValueError(f"Unsupported flavor_contents: {proc.flavor_contents}")

        # Legs for the two mesons
        meson1_fwd, meson1_bwd = flavor_map[f1]   # e.g. D meson
        meson2_fwd, meson2_bwd = flavor_map[f2]   # e.g. π meson

        # ------------------------------------------------------------------
        # 3. Interpolator builders — no signs here!
        # ------------------------------------------------------------------
        def phi_sink(op: BareOperator, t: int, mom: str):
            D = proc.get_elemental_block(mom, "disp")
            return contract("ij,ab->ijab", op.gamma, D[t], optimize="optimal")

        def phi_source(op: BareOperator, mom: str):
            D = proc.get_elemental_block(mom, "disp")
            return contract("ij,ab->ijab", op.gamma, D[0], optimize="optimal")

        # Pre-cache source interpolators
        phi0_m1_cache = [phi_source(op.op1, op.op1.mom) for op in op_list]  # D source
        phi0_m2_cache = [phi_source(op.op2, op.op2.mom) for op in op_list]  # π source

        # ------------------------------------------------------------------
        # 4. Single-meson correlators (local pseudoscalar)
        # ------------------------------------------------------------------
        first_op = op_list[0]
        phi0_m1_first = phi_source(first_op.op1, first_op.op1.mom)
        phi0_m2_first = phi_source(first_op.op2, first_op.op2.mom)

        meson1_corr = np.zeros((ntsrc, LT), dtype=np.cdouble)
        meson2_corr = np.zeros((ntsrc, LT), dtype=np.cdouble)

        for tsrc_idx in range(ntsrc):
            for t in range(LT):
                phi_t_1 = phi_sink(first_op.op1, t, first_op.op1.mom)
                phi_t_2 = phi_sink(first_op.op2, t, first_op.op2.mom)

                c1 = contract("ijab,jkbc,klcd,lida",
                            phi_t_1, meson1_fwd[tsrc_idx, t],
                            phi0_m1_first, meson1_bwd[tsrc_idx, t],
                            optimize="optimal")

                c2 = contract("ijab,jkbc,klcd,lida",
                            phi_t_2, meson2_fwd[tsrc_idx, t],
                            phi0_m2_first, meson2_bwd[tsrc_idx, t],
                            optimize="optimal")

                meson1_corr[tsrc_idx, t] = c1
                meson2_corr[tsrc_idx, t] = c2

        h5_group.create_dataset("meson1_correlator", data=meson1_corr.real)
        h5_group.create_dataset("meson2_correlator", data=meson2_corr.real)
        print("   → meson1 (D) and meson2 (π) correlators written")

        # ------------------------------------------------------------------
        # 5. Main di-meson matrix loop
        # ------------------------------------------------------------------
        for src_idx, src_op in enumerate(op_list):
            phi0_m1_src = phi0_m1_cache[src_idx]
            phi0_m2_src = phi0_m2_cache[src_idx]

            for snk_idx, snk_op in enumerate(op_list):
                direct   = np.zeros((ntsrc, LT), dtype=np.cdouble)
                crossing = np.zeros((ntsrc, LT), dtype=np.cdouble)

                for tsrc_idx in range(ntsrc):
                    for t in range(LT):
                        phi_t_1 = phi_sink(snk_op.op1, t, snk_op.op1.mom)
                        phi_t_2 = phi_sink(snk_op.op2, t, snk_op.op2.mom)

                        # Direct term: meson1 × meson2
                        m1 = contract("ijab,jkbc,klcd,lida",
                                    phi_t_1, meson1_fwd[tsrc_idx,t], phi0_m1_src,
                                    meson1_bwd[tsrc_idx,t], optimize="optimal")
                        m2 = contract("ijab,jkbc,klcd,lida",
                                    phi_t_2, meson2_fwd[tsrc_idx,t], phi0_m2_src,
                                    meson2_bwd[tsrc_idx,t], optimize="optimal")
                        direct[tsrc_idx, t] = m1 * m2

                        # Crossing term: swap backward legs
                        m1x = contract("ijab,jkbc,klcd,lida",
                                    phi_t_1, meson1_fwd[tsrc_idx,t], phi0_m1_src,
                                    meson2_bwd[tsrc_idx,t], optimize="optimal")
                        m2x = contract("ijab,jkbc,klcd,lida",
                                    phi_t_2, meson2_fwd[tsrc_idx,t], phi0_m2_src,
                                    meson1_bwd[tsrc_idx,t], optimize="optimal")
                        crossing[tsrc_idx, t] = m1x * m2x

                # Post-processing
                direct   = direct.real
                crossing = crossing.real

                if tsrc_avg:
                    for i in range(ntsrc):
                        sh = -tsrc_step * i
                        direct[i]   = np.roll(direct[i],   sh)
                        crossing[i] = np.roll(crossing[i], sh)
                    direct   = direct.mean(axis=0)
                    crossing = crossing.mean(axis=0)

                c15 = direct - crossing
                c6  = direct + crossing

                grp = h5_group.create_group(f"op{src_idx:02d}_op{snk_idx:02d}")
                grp.attrs["src"] = src_op.name
                grp.attrs["snk"] = snk_op.name
                grp.create_dataset("15", data=c15)
                grp.create_dataset("6",  data=c6)

        print(f"[DONE] Full matrix written in {(time.time()-start)/60:.1f} min")