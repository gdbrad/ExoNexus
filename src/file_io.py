# file_io.py
import os
import datetime
import numpy as np
import yaml
from typing import List, Dict, Tuple, Set, Any

from ingest_data import load_elemental, load_peram, reverse_perambulator_time


class DistillationObjectsIO:
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
        
        self.base_path = "/p/scratch/exflash/dpi-data"
        self.meson_path = "/p/scratch/exotichadrons/su3-distillation/data"

        self.dirs: Dict[str, str] = {
            "ens": os.path.join("/p/scratch/exflash/dpi-contractions", ens or ""),
            "light": os.path.join(self.base_path, ens or "", "perams_h5"),
            "meson": os.path.join(self.base_path, ens or "", "meson_h5"),
            "strange": os.path.join(self.base_path, ens or "", "perams_strange_sdb"), # only for su2
            "charm": os.path.join(self.base_path, ens or "", "perams_charm_h5"),
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
        self.peram_strange: np.ndarray | None = None # only for su2 
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
    # def _load_peram(self, flav: str, attr: str) -> None:
    #     p = self._file_path(flav)
    #     print(f"[IO] Trying {flav} perambulator: {p}")
    #     if not os.path.isfile(p):
    #         raise FileNotFoundError(f"Required {flav} perambulator missing: {p}")

    #     peram = load_peram(p, self.lt, self.nvecs, num_tsrcs=None, tsrc_step=self.tsrc_step)
    #     if peram is None:
    #         raise ValueError(f"load_peram returned None for {p}")
    #     setattr(self, attr, peram)
    #     print(f"[IO] {flav} perambulator loaded, shape={peram.shape}")

    def _load_peram(self, flav: str, attr: str) -> None:
        p = self._file_path(flav)
        print(f"[IO] Loading {flav} perambulator: {p}")
        peram = load_peram(
            p,
            max_t=self.lt,
            n_vecs=self.nvecs,
            num_tsrcs=self.ntsrc,           # ← AUTO-DETECT ALL
            tsrc_step=self.tsrc_step  
        )
        if peram is None:
            raise ValueError(f"Failed to load {flav} perambulator")
        setattr(self, attr, peram)
        print(f"    → loaded shape: {peram.shape}")

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
            raise RuntimeError("cfg_id not set - call get_contraction_params() first")
        if not self.flavor_contents:
            raise RuntimeError("flavor_contents not set- call get_contraction_params() first")

        # ---------- 1. meson elemental ----------
        # 1. Load FULL meson file
        self.meson_elemental_full = self._load_full_meson()
        print(f"[IO] full meson_elemental dataset loaded, shape={self.meson_elemental_full.shape}")

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

    # def perambulators(self) -> dict:
    #     """
    #     Select only charm perambulator slices whose physical time matches the light ones.
    #     This is the ONLY safe way when light and charm have different tsrc spacing.
    #     """
    #     light_fwd = self.peram_light   # shape (8, 64, ...)
    #     charm_fwd = self.peram_charm   # shape (16, 64, ...)

    #     # These are the actual physical times present in each file
    #     # Chroma names: t_source_0, t_source_4, t_source_8, ...
    #     light_times  = np.arange(0, 64, 8)[:light_fwd.shape[0]]   # → [0,8,16,24,32,40,48,56]
    #     charm_times  = np.arange(0, 64, 8)[:charm_fwd.shape[0]]   # → [0,4,8,...,60]

    #     print(f"[PERAM] Light sources at t = {light_times.tolist()}")
    #     print(f"[PERAM] Charm sources at t = {charm_times.tolist()}")

    #     # Find which charm indices correspond to light times
    #     charm_indices = []
    #     for t in light_times:
    #         matches = np.where(charm_times == t)[0]
    #         if len(matches) == 0:
    #             raise RuntimeError(f"Charm perambulator missing t={t} (needed for light source)")
    #         charm_indices.append(matches[0])

    #     charm_fwd_matched = charm_fwd[charm_indices]   # ← THIS IS THE KEY LINE

    #     print(f"[PERAM] Selected charm indices: {charm_indices}")
    #     print(f"[PERAM] Using {len(light_times)} PHYSICALLY IDENTICAL time sources")

    #     # Backward propagators with correct γ₅-hermiticity sign
    #     light_bwd = reverse_perambulator_time(light_fwd)
    #     charm_bwd = reverse_perambulator_time(charm_fwd_matched)

    #     self.ntsrc = len(light_times)

    #     return {
    #         "light_fwd": light_fwd,
    #         "light_bwd": light_bwd,
    #         "charm_fwd": charm_fwd_matched,
    #         "charm_bwd": charm_bwd,
    #     }

    # def perambulators(self):
    #     light_fwd = self.peram_light
    #     charm_fwd = self.peram_charm
    #     n_use = min(light_fwd.shape[0], charm_fwd.shape[0])
    #     print(f"[PERAM] Using {n_use} common sources (light: {light_fwd.shape[0]}, charm: {charm_fwd.shape[0]})")
        
    #     light_fwd = light_fwd[:n_use]
    #     charm_fwd = charm_fwd[:n_use]
        
    #     light_bwd = reverse_perambulator_time(light_fwd)
    #     charm_bwd = reverse_perambulator_time(charm_fwd)
        
    #     self.ntsrc = n_use
    #     return {"light_fwd": light_fwd, "light_bwd": light_bwd,
    #             "charm_fwd": charm_fwd, "charm_bwd": charm_bwd}

    def perambulators(self) -> dict:
        """
        Load ALL available time sources from disk, apply correct γ₅-hermiticity,
        and return only the common, valid ones.
        """
        if not hasattr(self, 'peram_light') or not hasattr(self, 'peram_charm'):
            raise RuntimeError("load_for_system() must be called first")

        # These were loaded with ntsrc=None → ALL sources from file
        light_fwd_all = self.peram_light   # shape (N_light, Lt, ...)
        charm_fwd_all = self.peram_charm   # shape (N_charm, Lt, ...)

        n_light = light_fwd_all.shape[0]
        n_charm = charm_fwd_all.shape[0]

        print(f"[PERAM] Light perambulator has {n_light} sources")
        print(f"[PERAM] Charm perambulator has {n_charm} sources")

        # Use ALL common sources — no arbitrary min with YAML
        n_use = min(n_light, n_charm)

        if n_use < n_light:
            print(f"    → Truncating light from {n_light} → {n_use} sources")
        if n_use < n_charm:
            print(f"    → Truncating charm from {n_charm} → {n_use} sources")

        # Slice to common set
        light_fwd = light_fwd_all[:n_use]
        charm_fwd = charm_fwd_all[:n_use]

        # Backward with correct sign
        light_bwd = reverse_perambulator_time(light_fwd)
        charm_bwd = reverse_perambulator_time(charm_fwd)

        # Override whatever was in YAML
        self.ntsrc = n_use

        print(f"[PERAM] Final: using {n_use} common time sources (all valid!)")
        print(f"    light_fwd shape : {light_fwd.shape}")
        print(f"    charm_fwd shape : {charm_fwd.shape}")

        return {
            "light_fwd": light_fwd,
            "light_bwd": light_bwd,
            "charm_fwd": charm_fwd,
            "charm_bwd": charm_bwd,
        }

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