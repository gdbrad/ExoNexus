import numpy as np 

from file_io import DistillationObjectsIO
from phi_factory import PhiFactory
from ingest_data import load_elemental, load_peram, reverse_perambulator_time


class DistillationData:

    def __init__(self, ens, cfg_id, collection=None):

        # IO layer
        self.io = DistillationObjectsIO(ens=ens, collection=collection)
        self.io.load_ens()

        # Config parameters
        self.cfg_id = cfg_id
        self.io.cfg_id = cfg_id

        self.nvecs = self.io.nvecs
        self.lt = self.io.lt
        self.ntsrc = self.io.ntsrc
        self.tsrc_step = self.io.tsrc_step
        self.flavor_contents = self.io.flavor_contents

        # Phi builder
        self.phi_builder = PhiFactory(self)

    # --------------------------------------------------
    # Loading
    # --------------------------------------------------

    def load_single_meson(self):

        # load perambulators
        for flav in self.flavor_contents:
            self.io.load_peram(flav)

        # load meson elementals
        self.io.load_meson()

    # --------------------------------------------------
    # Interfaces used by correlators
    # --------------------------------------------------

    def phi(self, op, t):
        return self.phi_builder.phi(op, t)

    def get_elemental_block(self, mom, disp):
        return self.io.get_elemental_block(mom, disp)

    def perambulators(self):
        """
        Select only charm perambulator slices whose physical time matches the light ones.
        This is the ONLY safe way when light and charm have different tsrc spacing.
        """
        light_fwd = self.peram_light   # shape (8, 64, ...)
        charm_fwd = self.peram_charm   # shape (16, 64, ...)
        
        # These are the actual physical times present in each file
        # Chroma names: t_source_0, t_source_4, t_source_8, ...
        light_times  = np.arange(0, self.lt, self.tsrc_step)[:light_fwd.shape[0]]   # → [0,8,16,24,32,40,48,56]
        charm_times  = np.arange(0, self.lt, self.tsrc_step)[:charm_fwd.shape[0]]   # → [0,4,8,...,60]

        print(f"[PERAM] Light sources at t = {light_times.tolist()}")
        print(f"[PERAM] Charm sources at t = {charm_times.tolist()}")

        # Find which charm indices correspond to light times
        charm_indices = []
        for t in light_times:
            matches = np.where(charm_times == t)[0]
            if len(matches) == 0:
                raise RuntimeError(f"Charm perambulator missing t={t} (needed for light source)")
            charm_indices.append(matches[0])

        charm_fwd_matched = charm_fwd[charm_indices]   # match light sources

        print(f"[PERAM] Selected charm indices: {charm_indices}")

        print(f"[PERAM] Using {len(light_times)} PHYSICALLY IDENTICAL time sources")

        light_bwd = reverse_perambulator_time(light_fwd)
        charm_bwd = reverse_perambulator_time(charm_fwd_matched)

        self.ntsrc = len(light_times)

        return {
            "light_fwd": light_fwd,
            "light_bwd": light_bwd,
            "charm_fwd": charm_fwd_matched,
            "charm_bwd": charm_bwd,
        }