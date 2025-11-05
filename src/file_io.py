
"""
distillation_processor.py

- DistillationObjectsIO      : I/O, path handling, loading of perambulators & elementals
- DistillationProcessor     : inherits I/O, adds correlator logic
  ├─ two_pt()                : orchestrator (class method)
  ├─ single_meson_correlator(): compute <M(t)M(0)> for one flavour
  └─ di_meson_correlator()   : compute direct / crossing / disconnected for two flavours
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any

import h5py
import numpy as np
import yaml
from insertion_factory import gamma

# ----------------------------------------------------------------------
# Package-specific imports
# ----------------------------------------------------------------------
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
from __init__ import (
    MESON_NAME_MAP,
    DI_MESON_NAME_MAP,
    BASE_PATH,
    TAPE_PATH,
)


# ----------------------------------------------------------------------
# Base I/O class
# ----------------------------------------------------------------------
class DistillationObjectsIO:
    def __init__(self, ens: str | None = None, collection: str | None = None) -> None:
        self.ens = ens
        if collection is None:
            collection = str(datetime.datetime.now())
            for c in " :.-":
                collection = collection.replace(c, "_")
        self.collection = collection

        self.dirs: Dict[str, str] = {
            "light": os.path.join(TAPE_PATH, ens, "perams_h5"),
            "meson": os.path.join(TAPE_PATH, ens, "meson_h5"),
            "strange": os.path.join(BASE_PATH, ens, "perams_strange_sdb"),
            "charm": os.path.join(BASE_PATH, ens, "perams_charm_sdb"),
        }

        # will be filled later
        self.nvecs: int | None = None
        self.lt: int | None = None
        self.ntsrc: int | None = None
        self.tsrc_step: int | None = None
        self.flavor_contents: List[str] = []
        self.cfg_id: int | None = None
        self.data1: bool = False

        # loaded data
        self.meson_elemental: Any | None = None
        self.peram_light: Any | None = None
        self.peram_strange: Any | None = None
        self.peram_charm: Any | None = None

    # ------------------------------------------------------------------
    # YAML → contraction parameters
    # ------------------------------------------------------------------
    def _get_contraction_settings(self) -> Dict[str, Any]:
        if self.ens is None:
            raise ValueError("Must specify ensemble!")
        yaml_path = Path(self.dirs["light"]).parent / f"{self.ens}.ini.yml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"No extraction input file: {yaml_path}")
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        return data[self.ens]

    def get_contraction_params(self) -> Dict[str, Any]:
        settings = self._get_contraction_settings()
        params = settings["params"]
        self.nvecs = params["nvecs"]
        self.lt = params["lt"]
        self.ntsrc = params["ntsrc"]
        self.tsrc_step = params["tsrc_step"]
        self.flavor_contents = params["flavor_contents"]
        return params.copy()

    # ------------------------------------------------------------------
    # File-spec helpers
    # ------------------------------------------------------------------
    def file_specs(self) -> Dict[str, Tuple[str, str]]:
        return {
            "light": (self.dirs["light"], f"peram_{self.nvecs}_cfg{{cfg_id}}.h5"),
            "meson": (self.dirs["meson"], f"meson-{self.nvecs}_cfg{{cfg_id}}.h5"),
            "strange": (self.dirs["strange"], f"peram_strange_nv{self.nvecs}_cfg{{cfg_id}}.h5"),
            "charm": (self.dirs["charm"], f"peram_charm_nv{self.nvecs}_cfg{{cfg_id}}.h5"),
        }

    def get_file_path(self, flavor: str = "meson") -> str | None:
        specs = self.file_specs()
        if flavor not in specs:
            print(f"Flavor '{flavor}' not recognised.")
            return None
        directory, tmpl = specs[flavor]
        path = Path(directory) / tmpl.format(cfg_id=self.cfg_id)
        return str(path) if path.is_file() else None

    # ------------------------------------------------------------------
    # Flavor logic
    # ------------------------------------------------------------------
    def _get_required_flavors(self) -> Set[str]:
        req: Set[str] = set()
        for fc in self.flavor_contents:
            if fc == "light_light":
                req.add("light")
            elif fc == "light_strange":
                req.update({"light", "strange"})
            elif fc == "light_charm":
                req.update({"light", "charm"})
            elif fc == "charm_strange":
                req.update({"charm", "strange"})
            elif fc == "charm_charm":
                req.add("charm")
            else:
                print(f"Unrecognised flavour content '{fc}', treating as light.")
                req.add("light")
        return req

    @property
    def required_flavors(self) -> Set[str]:
        return self._get_required_flavors()

    def get_meson_system_name(self) -> str:
        if len(self.flavor_contents) == 1:
            return MESON_NAME_MAP.get(self.flavor_contents[0], self.flavor_contents[0])
        pair = tuple(self.flavor_contents)
        return DI_MESON_NAME_MAP.get(pair, "_".join(self.flavor_contents))

    # ------------------------------------------------------------------
    # Load everything for a single cfg
    # ------------------------------------------------------------------
    def load_distillation_objects(self) -> None:
        if None in (self.nvecs, self.lt, self.ntsrc, self.tsrc_step, self.cfg_id):
            raise RuntimeError("Call get_contraction_params() first!")

        LT = self.lt
        nvecs = self.nvecs
        num_tsrc = self.ntsrc
        tsrc_step = self.tsrc_step

        # meson elemental
        meson_path = self.get_file_path("meson")
        if meson_path is None:
            self.meson_elemental = None
            return
        # TODO REPLACE MOM AND DISP IN TWO_PT_CORR FROM OPERATORY_FACTORY
        self.meson_elemental = load_elemental(meson_path, nvecs, mom="mom_0_0_0", disp="disp")

        # perambulators (only those needed)
        need_light = any("light" in fc for fc in self.flavor_contents)
        need_strange = any("strange" in fc for fc in self.flavor_contents)
        need_charm = any("charm" in fc for fc in self.flavor_contents)

        if need_light:
            p = self.get_file_path("light")
            if p:
                self.peram_light = load_peram(p, LT, nvecs, num_tsrc, tsrc_step)
        if need_strange:
            p = self.get_file_path("strange")
            if p:
                self.peram_strange = load_peram(p, LT, nvecs, num_tsrc, tsrc_step)
        if need_charm:
            p = self.get_file_path("charm")
            if p:
                self.peram_charm = load_peram(p, LT, nvecs, num_tsrc, tsrc_step)

    # ------------------------------------------------------------------
    # Build forward / backward map (used by both correlator methods)
    # ------------------------------------------------------------------
    def _peram_data(self) -> Dict[str, Tuple[Any, Any]]:
        data: Dict[str, Tuple[Any, Any]] = {}
        specs = self.file_specs()

        for fc in self.flavor_contents:
            if fc == "light_light":
                peram = self.peram_light
                back = reverse_perambulator_time(self.peram_light)
                files = (specs["light"][0], specs["light"][0])
            elif fc == "light_strange":
                peram = self.peram_light
                back = reverse_perambulator_time(self.peram_strange)
                files = (specs["light"][0], specs["strange"][0])
            elif fc == "light_charm":
                peram = self.peram_light
                back = reverse_perambulator_time(self.peram_charm)
                files = (specs["light"][0], specs["charm"][0])
            elif fc == "charm_strange":
                peram = self.peram_charm
                back = reverse_perambulator_time(self.peram_strange)
                files = (specs["charm"][0], specs["strange"][0])
            elif fc == "charm_charm":
                peram = self.peram_charm
                back = reverse_perambulator_time(self.peram_charm)
                files = (specs["charm"][0], specs["charm"][0])
            else:
                peram = self.peram_light
                back = reverse_perambulator_time(self.peram_light)
                files = (specs["light"][0], specs["light"][0])

            if peram is None or back is None:
                raise FileNotFoundError(f"Missing perambulator for {fc}")

            data[fc] = (peram, back)
            print(f"Loaded {fc}: forward {files[0]}, backward {files[1]}")
        return data