from typing import List, Dict, Tuple, Any
import h5py
import numpy as np
from insertion_factory import gamma
import file_io
#from operator_factory import MHI, MHICollection
from dimeson_factory import DiMesonOperator
import re
from ingest_data import load_elemental, reverse_perambulator_time
from itertools import product
from opt_einsum import contract  # Assuming we keep it, but can replace with np.einsum if needed

# ----------------------------------------------------------------------
# Processor – correlator logic
# ----------------------------------------------------------------------
class CorrelatorFactory(file_io.DistillationObjectsIO):
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
        self.load_distillation_objects()

    # ------------------------------------------------------------------
    # Contract functions integrated as methods
    # ------------------------------------------------------------------
    def contract_local(self, operator, t: int, mom: str):
        D = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp')
        phi = contract("ij,ab->ijab", operator.gamma, D[t], optimize="optimal")
        return phi

    def contract_nabla(self, operator, t: int, mom: str):
        gamma_i = [gamma.gamma[1], gamma.gamma[2], gamma.gamma[3]]
        D1 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_1')
        D2 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_2')
        D3 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_3')
        D = [D1, D2, D3]
        phi = sum(
            contract("ij,ab->ijab", operator.gamma @ gamma_i[i], D[i][t], optimize="optimal")
            for i in range(3)
        )
        return phi

    def contract_B_D(self, operator, t: int, mom: str, add: bool = True):
        coeff = 1 if add else -1
        # load elementals displaced with two covariant derivatives
        D1D2 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_1_2')
        D2D1 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_2_1')
        D1D3 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_1_3')
        D3D1 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_3_1')
        D2D3 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_2_3')
        D3D2 = load_elemental(self.get_file_path("meson"), self.lt, self.nvecs, mom=mom, disp='disp_3_2')
        # terms at given t
        phi_1 = contract("ij,ab->ijab", gamma.gamma[1], D2D3[t], optimize="optimal")
        phi_2 = contract("ij,ab->ijab", gamma.gamma[1], D3D2[t], optimize="optimal")  # Subtract this one if coeff=-1
        phi_3 = contract("ij,ab->ijab", gamma.gamma[2], D3D1[t], optimize="optimal")
        phi_4 = contract("ij,ab->ijab", gamma.gamma[2], D1D3[t], optimize="optimal")  # Subtract this one
        phi_5 = contract("ij,ab->ijab", gamma.gamma[3], D1D2[t], optimize="optimal")
        phi_6 = contract("ij,ab->ijab", gamma.gamma[3], D2D1[t], optimize="optimal")  # Subtract this one
        phi = (phi_1 - coeff * phi_2 +
               phi_3 - coeff * phi_4 +
               phi_5 - coeff * phi_6)
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
        """Entry point - decides which sub-method to call."""
        if proc.meson_elemental is None:
            print(f"Meson elemental missing for cfg {proc.cfg_id}")
            return False
        LT = proc.lt
        num_tsrc = proc.ntsrc
        tsrc_step = proc.tsrc_step
        phi_0 = np.einsum(
            "ij,ab->ijab", gamma.gamma[5], proc.meson_elemental[0], optimize="optimal"
        )
        peram_data = proc._peram_data()

        # ------------------------------------------------------------------
        # 1. Di-meson correlator matrix case
        # ------------------------------------------------------------------
        if len(proc.flavor_contents) == 2:
            proc.di_meson_correlator_matrix(
                operators=operators,
                h5_group=h5_group,
                peram_data=peram_data,
                phi_0=phi_0,
                tsrc_avg=tsrc_avg,
                three_bar=three_bar,
            )
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 1. Di-meson case
        # ------------------------------------------------------------------
        if len(proc.flavor_contents) == 2:
            proc.di_meson_correlator(
                h5_group=h5_group,
                peram_data=peram_data,
                phi_0=phi_0,
                tsrc_avg=tsrc_avg,
                three_bar=three_bar,
            )
        # ------------------------------------------------------------------
        # 3. Single-meson case
        # ------------------------------------------------------------------
        else:
            proc.single_meson_correlator(
                h5_group=h5_group,
                peram_data=peram_data,
                phi_0=phi_0,
                tsrc_avg=tsrc_avg,
            )
        # ------------------------------------------------------------------
        # 3. Always write the *individual* meson correlators for every flavour
        # ------------------------------------------------------------------
        for idx, fc in enumerate(proc.flavor_contents, 1):
            peram, peram_b = peram_data[fc]
            single = np.zeros((num_tsrc, LT), dtype=np.cdouble)
            for tsrc_idx in range(num_tsrc):
                for dt in range(LT):
                    phi_t = np.einsum(
                        "ij,ab->ijab",
                        gamma.gamma[5],
                        proc.meson_elemental[dt],
                        optimize="optimal",
                    )
                    tau = peram[tsrc_idx, dt, :, :, :, :]
                    tau_b = peram_b[tsrc_idx, dt, :, :, :, :]
                    single[tsrc_idx, dt] = np.einsum(
                        "ijab,jkbc,klcd,lida",
                        phi_t,
                        tau,
                        phi_0,
                        tau_b,
                        optimize="optimal",
                    )
            single = single.real
            if tsrc_avg:
                for i in range(num_tsrc):
                    single[i] = np.roll(single[i], -tsrc_step * i)
                single = single.mean(axis=0)
            key = f"meson{idx}_{fc}"
            suffix = "_tsrc_avg" if tsrc_avg else ""
            h5_group.create_dataset(f"{key}/cfg_{proc.cfg_id}{suffix}", data=single)
        print(f"Cfg {proc.cfg_id} processed – results written.")
        return True

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
                phi_t = np.einsum(
                    "ij,ab->ijab",
                    gamma.gamma[5],
                    proc.meson_elemental[dt],
                    optimize="optimal",
                )
                tau = peram[tsrc_idx, dt, :, :, :, :]
                tau_b = peram_b[tsrc_idx, dt, :, :, :, :]
                data[tsrc_idx, dt] = np.einsum(
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
                phi_t = np.einsum(
                    "ij,ab->ijab",
                    gamma.gamma[5],
                    proc.meson_elemental[t],
                    optimize="optimal",
                )
                # direct legs
                tau1 = p1[tsrc_idx, t, :, :, :, :]
                tau1b = p1b[tsrc_idx, t, :, :, :, :]
                m1 = np.einsum(
                    "ijab,jkbc,klcd,lida",
                    phi_t,
                    tau1,
                    phi_0,
                    tau1b,
                    optimize="optimal",
                )
                tau2 = p2[tsrc_idx, t, :, :, :, :]
                tau2b = p2b[tsrc_idx, t, :, :, :, :]
                m2 = np.einsum(
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
                    m1c = np.einsum(
                        "ijab,jkbc,klcd,lida",
                        phi_t,
                        tau1,
                        phi_0,
                        p2b[tsrc_idx, t, :, :, :, :],
                        optimize="optimal",
                    )
                    m2c = np.einsum(
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
                    loop = np.einsum(
                        "siab,sjcd->abcd",
                        p2[tsrc_idx, t, :, :, :, :],
                        p2b[tsrc_idx, t, :, :, :, :],
                        optimize="optimal",
                    )
                    disc[tsrc_idx, t] = m1 * np.einsum(
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
        irrep: str,
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
        num_op = len(operators)
        direct = np.zeros((num_op, num_op, num_tsrc, LT), dtype=np.cdouble)
        crossing = np.zeros((num_op, num_op, num_tsrc, LT), dtype=np.cdouble) if not identical else None
        disc = np.zeros((num_op, num_op, num_tsrc, LT), dtype=np.cdouble) if three_bar else None
        p1, p1b = peram_data[f1]
        p2, p2b = peram_data[f2]
        for src_idx, (src_name, src_op) in enumerate(operators.items()):
            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                for tsrc_idx in range(num_tsrc):
                    for t in range(LT):
                        # Compute phi_0 based on src_op
                        if src_op.deriv is None:
                            phi_0 = proc.contract_local(src_op, 0, src_op.mom)
                        elif src_op.deriv == "nabla":
                            phi_0 = proc.contract_nabla(src_op, 0, src_op.mom)
                        elif src_op.deriv in ["B", "D"]:
                            phi_0 = proc.contract_B_D(src_op, 0, src_op.mom, add=(src_op.deriv == "D"))
                        else:
                            continue
                        # Compute phi_t based on snk_op
                        if snk_op.deriv is None:
                            phi_t = proc.contract_local(snk_op, t, snk_op.mom)
                        elif snk_op.deriv == "nabla":
                            phi_t = proc.contract_nabla(snk_op, t, snk_op.mom)
                        elif snk_op.deriv in ["B", "D"]:
                            phi_t = proc.contract_B_D(snk_op, t, snk_op.mom, add=(snk_op.deriv == "D"))
                        else:
                            continue
                        # direct legs
                        tau1 = p1[tsrc_idx, t, :, :, :, :]
                        tau1b = p1b[tsrc_idx, t, :, :, :, :]
                        m1 = np.einsum(
                            "ijab,jkbc,klcd,lida",
                            phi_t,
                            tau1,
                            phi_0,
                            tau1b,
                            optimize="optimal",
                        )
                        tau2 = p2[tsrc_idx, t, :, :, :, :]
                        tau2b = p2b[tsrc_idx, t, :, :, :, :]
                        m2 = np.einsum(
                            "ijab,jkbc,klcd,lida",
                            phi_t,
                            tau2,
                            phi_0,
                            tau2b,
                            optimize="optimal",
                        )
                        direct[src_idx, snk_idx, tsrc_idx, t] = m1 * m2
                        # crossing
                        if not identical:
                            m1c = np.einsum(
                                "ijab,jkbc,klcd,lida",
                                phi_t,
                                tau1,
                                phi_0,
                                p2b[tsrc_idx, t, :, :, :, :],
                                optimize="optimal",
                            )
                            m2c = np.einsum(
                                "ijab,jkbc,klcd,lida",
                                phi_t,
                                tau2,
                                phi_0,
                                p1b[tsrc_idx, t, :, :, :, :],
                                optimize="optimal",
                            )
                            crossing[src_idx, snk_idx, tsrc_idx, t] = m1c * m2c
                        # disconnected loop for bar{3}
                        if three_bar:
                            loop = np.einsum(
                                "siab,sjcd->abcd",
                                p2[tsrc_idx, t, :, :, :, :],
                                p2b[tsrc_idx, t, :, :, :, :],
                                optimize="optimal",
                            )
                            disc[src_idx, snk_idx, tsrc_idx, t] = m1 * np.einsum(
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
                for op1 in range(num_op):
                    for op2 in range(num_op):
                        direct[op1, op2, i] = np.roll(direct[op1, op2, i], shift)
                        if not identical:
                            crossing[op1, op2, i] = np.roll(crossing[op1, op2, i], shift)
                        if three_bar:
                            disc[op1, op2, i] = np.roll(disc[op1, op2, i], shift)
            direct = direct.mean(axis=2)
            crossing = crossing.mean(axis=2) if not identical else None
            disc = disc.mean(axis=2) if three_bar else None
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


class StubProc(CorrelatorFactory):
    # Interface with correlator_matrix: Create dict of operators keyed by name
    #operators: Dict[str, DiMesonOperator] = {dim.name: dim for dim in di_mesons}
    #print(f"Created operators dict with {len(operators)} entries.")
    flavor_contents = ['charm_light', 'light_light']
    lt = 64
    ntsrc = 4
    tsrc_step = 16
    cfg_id = 0
    def get_meson_system_name(self):
        return 'D_pi'

# Stub peram_data and h5_group
peram_data = {
    'charm_light': (np.random.rand(4, 64, 4, 4, 4, 4), np.random.rand(4, 64, 4, 4, 4, 4)),
    'light_light': (np.random.rand(4, 64, 4, 4, 4, 4), np.random.rand(4, 64, 4, 4, 4, 4))
}
h5_group = None  # In real: h5py.Group

# Call the method (note: method uses src_op/snk_op as BareOperator, but passes DiMesonOperator.
# FIXED in method body above by using src_op.op1/op2 for contractions, but adjust loops if needed for per-meson deriv/gamma.
# For now, assuming placeholder phi; implement proc.contract_* for op1/op2 separately if deriv differs.)
operators = DiMesonOperator.generate_operators()
# Then call cls.di_meson_correlator_matrix(operators=operators, irrep='a1p', proc=instance, ...)

StubProc.di_meson_correlator_matrix(
    operators=operators,
    irrep='a1p',
    proc=StubProc(),  # Or cls if static
    h5_group=h5_group,
    peram_data=peram_data,
    phi_0=None,  # Unused
    tsrc_avg=True,
    three_bar=False
)