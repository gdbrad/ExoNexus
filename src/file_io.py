# file_io.py
import os
import datetime
import h5py
import numpy as np
import yaml
from typing import List, Dict, Tuple, Set, Any

from insertion_factory import gamma
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
from __init__ import MESON_NAME_MAP, DI_MESON_NAME_MAP


class DistillationObjectsIO:
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, ens: str | None = None, collection: str | None = None) -> None:
        self.ens = ens
        self.collection = (
            collection
            or str(datetime.datetime.now())
            .replace(" ", "_")
            .replace(":", "_")
            .replace(".", "_")
            .replace("-", "_")
        )
        self.base_path = "/p/scratch/exflash/su3-distillation-juwels"

        self.dirs: Dict[str, str] = {
            "ens": os.path.join("/p/scratch/exflash/dpi-contractions", ens or ""),
            "light": os.path.join(self.base_path, ens or "", "perams_h5"),
            "meson": os.path.join(self.base_path, ens or "", "meson_h5"),
            "strange": os.path.join(self.base_path, ens or "", "perams_strange_sdb"),
            "charm": os.path.join(self.base_path, ens or "", "perams_charm_sdb"),
        }

        # will be filled by get_contraction_params()
        self.nvecs: int | None = None
        self.lt: int | None = None
        self.ntsrc: int | None = None
        self.tsrc_step: int | None = None
        self.flavor_contents: List[str] = []
        self.cfg_id: int | None = None

        # loaded objects – **always an ndarray or raise**
        self.meson_elemental: np.ndarray | None = None
        self.peram_light: np.ndarray | None = None
        self.peram_strange: np.ndarray | None = None
        self.peram_charm: np.ndarray | None = None

        # Full meson file: (mom, disp, t, i, j)
        self.meson_elemental_full: np.ndarray | None = None

        # Cache: (mom, disp) → (t, i, j) block
        self._elemental_cache: Dict[Tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # YAML → contraction parameters
    # ------------------------------------------------------------------
    def _get_contraction_settings(self) -> Dict[str, Any]:
        if not self.ens:
            raise ValueError("Ensemble name required")
        yaml_path = os.path.join(self.dirs["ens"], f"{self.ens}.ini.yml")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML config missing: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return data[self.ens]

    def get_contraction_params(self) -> Dict[str, Any]:
        s = self._get_contraction_settings()
        p = s["params"]
        self.nvecs = p["nvecs"]
        self.lt = p["lt"]
        self.ntsrc = p["ntsrc"]
        self.tsrc_step = p["tsrc_step"]
        self.flavor_contents = p["flavor_contents"]
        return p.copy()

    # ------------------------------------------------------------------
    # Helper – full path for a flavor
    # ------------------------------------------------------------------
    def _file_path(self, flavor: str) -> str:
        tmpl = {
            "light": f"peram_{self.nvecs}_cfg{self.cfg_id}.h5",
            "meson": f"meson-{self.nvecs}_cfg{self.cfg_id}.h5",
            "strange": f"peram_strange_nv{self.nvecs}_cfg{self.cfg_id}.h5",
            "charm": f"peram_charm_{self.nvecs}_cfg{self.cfg_id}.h5",
        }[flavor]
        return os.path.join(self.dirs[flavor], tmpl)
    
    # ------------------------------------------------------------------
    # Load FULL meson file (all mom, all disp)
    # ------------------------------------------------------------------
    def _load_full_meson(self) -> np.ndarray:
        path = self._file_path("meson")
        print(f"[IO] Loading FULL meson file: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Meson file missing: {path}")

        # This assumes load_elemental can return *all* data when mom/disp=None
        full = load_elemental(path, max_t=self.lt, n_vecs=self.nvecs, mom=None, disp=None)
        if full is None:
            raise ValueError("load_elemental returned None for full meson file")
        print(f"[IO] Full meson loaded, shape={full.shape}")
        return full
    
    # ------------------------------------------------------------------
    # Extract (mom, disp) block ON DEMAND from full array
    # ------------------------------------------------------------------
    def get_elemental_block(self, mom: str, disp: str) -> np.ndarray:
        key = (mom, disp)
        if key in self._elemental_cache:
            return self._elemental_cache[key]

        if self.meson_elemental_full is None:
            raise RuntimeError("Full meson file not loaded. Call load_for_system() first.")

        # You need to know the **layout** of your HDF5 file.
        # Assuming: dataset name = f"{mom}/{disp}" or similar.
        # We'll use a **fallback**: let load_elemental handle indexing.
        block = load_elemental(
            self._file_path("meson"),
            max_t=self.lt,
            n_vecs=self.nvecs,
            mom=mom,
            disp=disp,
        )
        if block is None:
            raise ValueError(f"Failed to extract {mom}/{disp}")
        self._elemental_cache[key] = block
        return block

    # ------------------------------------------------------------------
    # Load a *single* elemental block (mom + disp) on demand
    # ------------------------------------------------------------------
    # def _load_elemental_block(self) -> np.ndarray:
    #     # key = (mom, disp)
    #     # if key in self._elemental_cache:
    #     #     return self._elemental_cache[key]

    #     path = self._file_path("meson")
    #     if not os.path.isfile(path):
    #         raise FileNotFoundError(f"Meson file missing: {path}")

    #     arr = load_elemental(
    #         path,
    #         max_t=self.lt,
    #         n_vecs=self.nvecs,
    #         # mom=mom,
    #         # disp=disp,
    #     )
    #     # if arr is None:
    #     #     raise ValueError(f"load_elemental returned None for mom={mom}, disp={disp}")
    #     # self._elemental_cache[key] = arr
    #     return arr

    # ------------------------------------------------------------------
    # Helper – load a single perambulator (class method)
    # ------------------------------------------------------------------
    def _load_peram(self, flav: str, attr: str) -> None:
        p = self._file_path(flav)
        print(f"[IO] Trying {flav} perambulator: {p}")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required {flav} perambulator missing: {p}")

        peram = load_peram(p, self.lt, self.nvecs, self.ntsrc, self.tsrc_step)
        if peram is None:
            raise ValueError(f"load_peram returned None for {p}")
        setattr(self, attr, peram)
        print(f"[IO] {flav} perambulator loaded, shape={peram.shape}")

    # ------------------------------------------------------------------
    # PUBLIC: load everything for a di-meson system (e.g. "Dpi")
    # ------------------------------------------------------------------
    def load_for_system(self, system_name: str) -> None:
        """
        Load meson elemental + *all* perambulators that belong to the
        requested system. Raises a clear FileNotFoundError if a required
        perambulator is missing.
        """
        if self.cfg_id is None:
            raise RuntimeError("cfg_id not set – call get_contraction_params() first")
        if not self.flavor_contents:
            raise RuntimeError("flavor_contents not set – call get_contraction_params() first")

        # ---------- 1. meson elemental ----------
        # 1. Load FULL meson file
        self.meson_elemental_full = self._load_full_meson()
        # print(f"[IO] full meson_elemental dataset loaded, shape={self.meson_elemental_full.shape}")

        # ---------- 2. which perambulators are needed? ----------
        required: Set[str] = set()
        for fc in self.flavor_contents:
            if "light" in fc:
                required.add("light")
            if "strange" in fc:
                required.add("strange")
            if "charm" in fc:
                required.add("charm")

        # ---------- 3. load each required perambulator ----------
        for flav in required:
            if flav == "light":
                self._load_peram("light", "peram_light")
            elif flav == "strange":
                self._load_peram("strange", "peram_strange")
            elif flav == "charm":
                self._load_peram("charm", "peram_charm")

    # ------------------------------------------------------------------
    # Build forward / backward perambulator map
    # ------------------------------------------------------------------
    def _peram_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Return a dictionary mapping flavor combination → (forward, backward) perambulators.
        The backward perambulator is the time-reversed version.
        """
        data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for fc in self.flavor_contents:
            # ----- decide forward perambulator -----
            if fc == "light_light":
                forward = self.peram_light
                reverse_from = self.peram_light
            elif fc in ("light_charm", "charm_light"):
                forward = self.peram_light
                reverse_from = self.peram_charm
            elif fc == "charm_charm":
                forward = self.peram_charm
                reverse_from = self.peram_charm
            else:
                raise ValueError(f"Unsupported flavor combination: {fc}")

            # ----- sanity check -----
            if forward is None:
                raise RuntimeError(f"Forward perambulator for {fc} is None – was load_for_system() called?")
            if reverse_from is None:
                raise RuntimeError(f"Reverse source perambulator for {fc} is None – was load_for_system() called?")

            # ----- build backward perambulator -----
            backward = reverse_perambulator_time(reverse_from)

            data[fc] = (forward, backward)

        return data