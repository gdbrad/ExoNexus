from typing import List, Dict, Tuple, Any

import h5py
import numpy as np
from insertion_factory import gamma
import file_io

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
        tsrc_step: int = 1,
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
        # 2. Single-meson case
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
    # Di-meson correlator
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
