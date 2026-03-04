def load_ens(self) -> Dict[str, Any]:
    if not self.ens:
        raise ValueError("Ensemble name required")
    yaml_path = self.yaml_file
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML config missing: {yaml_path}")
    with open(yaml_path, "r") as f:
        full_cfg = yaml.safe_load(f)

    if self.ens not in full_cfg:
        raise ValueError(f"Ensemble {self.ens} not found in YAML")
    
    cfg = full_cfg[self.ens]  # now cfg points to the nested dict for the ensemble

    # -------------------- Paths --------------------
    raw_paths = cfg["paths"]
    self.base_path = raw_paths.get("base_path")
    self.driver_path = raw_paths.get("driver_path")
    self.venv_activate = raw_paths.get("venv_activate")
    self.correlator_output = raw_paths.get("correlator_output")

    self.paths = {k: v for k, v in raw_paths.items()}

    # -------------------- SLURM --------------------
    self.slurm_cfg = cfg.get("slurm", {})
    self.output_dir = self.slurm_cfg.get("output_dir", "")
    self.log_dir = self.slurm_cfg.get("log_dir", "")

    # -------------------- Config IDs --------------------
    r = cfg["configs"]["range"]
    self.config_ids = list(range(r["start"], r["end"] + 1, r.get("step", 1)))
    exclude = set(cfg["configs"].get("exclude", []))
    self.config_ids = [c for c in self.config_ids if c not in exclude]

    # -------------------- Distillation --------------------
    p = cfg["distillation_input"]
    self.nvecs = p["nvecs"]
    self.lt = p["lt"]
    self.ntsrc = p["ntsrc"]
    self.tsrc_step = p["tsrc_step"]

    # -------------------- Flavors --------------------
    self.flavor_contents = cfg.get("flavors", [])

    # -------------------- Filename templates --------------------
    # You may have templates in the YAML, optional
    self.filename_templates = cfg.get("filenames", {})

    return cfg