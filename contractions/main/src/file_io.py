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
            full_cfg = yaml.safe_load(f)

        # Scope settings to the ensemble key if nested
        cfg = full_cfg.get(self.ens, full_cfg)

        raw_paths = cfg.get("paths", {})
        self.base_path = raw_paths.get("base_path", "")
        self.data_path = raw_paths.get("data_path", self.base_path)

        # Setup default fallback paths for distillation objects
        self.paths = cfg.get("data_dirs", {
            "meson": os.path.join(self.data_path, "meson_sdb", f"numvec64"),
            "light": os.path.join(self.data_path, "perams_sdb", f"numvec64"),
            "charm": os.path.join(self.data_path, "perams_charm_sdb", f"numvec64"),
            "strange": os.path.join(self.data_path, "perams_strange_sdb")
        })
        
        # Use filenames from config if present, else fallback to standard Chroma format
        self.filename_templates = cfg.get("filenames", {
            "meson": "meson-{nvecs}_cfg{cfg_id}.h5",
            "light": "peram_{nvecs}_cfg{cfg_id}.h5",
            "charm": "peram_charm_{nvecs}_cfg{cfg_id}.h5",
            "strange": "peram_strange_{nvecs}_cfg{cfg_id}.h5"
        })

        # Configuration indices
        r = cfg.get("configs", {}).get("range", {})
        if r:
            self.config_ids = list(range(r.get("start", 0), r.get("end", 0) + 1, r.get("step", 1)))
        else:
            self.config_ids = []

        exclude = set(cfg.get("configs", {}).get("exclude", []))
        self.config_ids = [c for c in self.config_ids if c not in exclude]
        
        # Distillation parameters
        p = cfg.get("distillation_input", {})
        self.nvecs = p.get("nvecs")
        self.lt = p.get("lt")
        self.ntsrc = p.get("ntsrc")
        self.tsrc_step = p.get("tsrc_step")

        self.flavor_contents = cfg.get("flavors", [])
        return cfg

    def _file_path(self, flavor: str) -> str:
        if flavor not in self.filename_templates:
            raise ValueError(f"No filename template defined for flavor '{flavor}'")
        
        filename = self.filename_templates[flavor].format(
            nvecs=self.nvecs,
            cfg_id=self.cfg_id,
        )
        base_dir = self.paths.get(flavor, "")
        return os.path.join(base_dir, filename)

    ################ meson loading and caching ###################################

    # load all mom and disp
    def load_full_meson(self) -> np.ndarray:
        path = self._file_path("meson")
        print(f"[IO] Loading FULL meson file: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Meson file missing: {path}")

        # assumes load_elemental can return *all* data when mom/disp=None
        _data,_ = load_elemental(
            path, 
            # max_t=self.lt, 
            # n_vecs=self.nvecs,
            mom=None,
            disp=None)
        self.meson_full = _data
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
        block,_ = load_elemental(
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
        
        peram,metadata = load_peram(
            flav_path
            # max_t=self.lt,
            # n_vecs=self.nvecs,
            # num_tsrcs=self.ntsrc,           
            # tsrc_step=self.tsrc_step  
        )
        if peram is None:
            raise ValueError(f"Failed to load {flav} perambulator")
        
        self.perams[flav] = peram