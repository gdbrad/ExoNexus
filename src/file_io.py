import os
import datetime
import numpy as np
import yaml
from typing import List, Dict, Tuple, Set, Any

from ingest_data import load_elemental, load_peram, reverse_perambulator_time

class DistillationObjectsIO:
    def __init__(self, 
                 ens: str,
                 yaml_file: str, 
                 collection: str | None = None) -> None:
        self.ens = ens
        self.yaml_file = yaml_file
        self.collection = (
            collection
            or str(datetime.datetime.now())
            .replace(" ", "_")
            .replace(":", "_")
            .replace(".", "_")
            .replace("-", "_")
        )

        # will be filled by get_contraction_params()
        self.nvecs = None
        self.lt = None
        self.ntsrc = None
        self.tsrc_step = None
        self.flavor_contents = []
        
        self.dirs = {}
        self._filename_templates = {}

        self.cfg_id = None

        # loaded objects - always ndarray
        # Full meson file: (mom, disp, t, i, j)
        self.meson_full: np.ndarray | None = None
        self.perams: Dict[str, np.ndarray] = {}
        # Cache: (mom, disp) → (t, i, j) block
        self._elemental_cache: Dict[Tuple[str, str], np.ndarray] = {}

    # YAML → contraction parameters
    def load_ens(self) -> Dict[str, Any]:
        if not self.ens:
            raise ValueError("Ensemble name required")
        yaml_path = self.yaml_file
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML config missing: {yaml_path}")
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)[self.ens]

        raw_paths = cfg["paths"]
        # First resolve base paths
        self.tape_path = raw_paths["tape_path"]
        self.data_path = raw_paths["data_path"]
        # Now expand templated paths
        self.paths = {}
        for key, val in raw_paths.items():
            self.paths[key] = val.format(data_path=self.data_path)

        self.filename_templates = cfg["filenames"]
        # --------------------------------------------------
        self.output_dir = cfg["slurm"]["output_dir"]
        self.log_dir = cfg["slurm"]["log_dir"]
        # --------------------------------------------------
        r = cfg["configs"]["range"]
        self.config_ids = list(
            range(r["start"], r["end"] + 1, r["step"])
        )

        exclude = set(cfg["configs"]["exclude"])
        self.config_ids = [c for c in self.config_ids if c not in exclude]
        
        # Distillation parameters
        p = cfg["params"]
        self.num_configs = p["num_configs"]
        self.nvecs = p["nvecs"]
        self.lt = p["lt"]
        self.ntsrc = p["ntsrc"]
        self.tsrc_step = p["tsrc_step"]
        self.flavor_contents = p["flavor_contents"]
        self.system = cfg["system"]

    def _file_path(self, flavor: str) -> str:
        if flavor not in self.filename_templates:
            raise ValueError(f"No filename template defined for flavor '{flavor}'")
        filename = self.filename_templates[flavor].format(
            nvecs=self.nvecs,
            cfg_id=self.cfg_id,
        )
        base_path = self.paths[flavor]
        return os.path.join(base_path, filename)

    ################ meson loading and caching ###################################

    # load all mom and disp
    def load_full_meson(self) -> np.ndarray:
        path = self._file_path("meson")
        print(f"[IO] Loading FULL meson file: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Meson file missing: {path}")

        # assumes load_elemental can return *all* data when mom/disp=None
        self.meson_full = load_elemental(
            path, 
            # max_t=self.lt, 
            # n_vecs=self.nvecs,
            mom=None,
            disp=None)
        
        if self.meson_full is None:
            raise ValueError("failed to load full meson file")
        
        print(f"[IO] Full meson loaded, shape={self.meson_full.shape}")
    
    def get_elemental_block(self, mom: str, disp: str) -> np.ndarray:
        """
        get (mom, disp) block on demand from full array
        """
        key = (mom, disp)
        if key in self._elemental_cache:
            return self._elemental_cache[key]

        # this assumes dataset name = f"{mom}/{disp}"
        block = load_elemental(
            self._file_path("meson"),
            mom=mom,
            disp=disp,
        )
        if block is None:
            raise ValueError(f"failed to get elemental block {mom}/{disp}")
        self._elemental_cache[key] = block
        return block

    ###################### peram loading ######################################

    def load_peram(self, flav: str):

        flav_path = self._file_path(flav)
        print(f"[IO] Loading {flav} perambulator: {flav_path} for cfg {self.cfg_id}")
        
        peram = load_peram(
            flav_path
            # max_t=self.lt,
            # n_vecs=self.nvecs,
            # num_tsrcs=self.ntsrc,           
            # tsrc_step=self.tsrc_step  
        )
        if peram is None:
            raise ValueError(f"Failed to load {flav} perambulator")
        
        self.perams[flav] = peram