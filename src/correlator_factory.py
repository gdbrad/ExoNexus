import numpy as np
from typing import Any

from file_io import DistillationObjectsIO
from elemental_factory import ElementalFactory


class CorrelatorFactory(DistillationObjectsIO):
    """
    Thin wrapper around DistillationObjectsIO.

    Responsibilities:
        - Load YAML contraction parameters
        - Set cfg_id
        - Load required perambulators + elementals
        - Provide phi() interface
        - Provide perambulator dictionary
    """

    # -----------------------------------------------------
    # Constructor
    # -----------------------------------------------------

    def __init__(
        self,
        ens: str,
        cfg_id: int,
    ) -> None:

        super().__init__(ens=ens)

        self.cfg_id = cfg_id

        # Load contraction parameters from YAML
        params = self.get_contraction_params()

        self.nvecs = params["nvecs"]
        self.lt = params["lt"]
        self.ntsrc = params["ntsrc"]
        self.tsrc_step = params["tsrc_step"]
        self.flavor_contents = params["flavor_contents"]

    # -----------------------------------------------------
    # Public loader
    # -----------------------------------------------------

    def load(self, system_name: str = "single") -> None:
        """
        Load elementals + required perambulators.

        system_name is passed through to IO layer.
        """
        self.load_for_system(system_name)

    # -----------------------------------------------------
    # Distillation operator Φ
    # -----------------------------------------------------

    def phi(self, operator: Any, t: int) -> np.ndarray:
        """
        Build Φ(t) for a given operator.

        Expects operator to contain:
            - momentum
            - displacement
            - gamma structure
        """

        # operator expected to be a dict-like structure
        mom = operator["mom"]
        disp = operator["disp"]
        gamma = operator["gamma"]

        # Get (t, i, j) block for that momentum/displacement
        elemental = self.get_elemental_block(mom, disp)

        # build_phi handles gamma insertion + spin structure
        return (
            elemental=elemental,
            gamma=gamma,
            t=t,
            nvecs=self.nvecs,
        )

    # -----------------------------------------------------
    # Forward/backward perambulators
    # -----------------------------------------------------

    def perambulators(self) -> dict:
        """
        Direct passthrough to IO implementation.
        """
        return super().perambulators()