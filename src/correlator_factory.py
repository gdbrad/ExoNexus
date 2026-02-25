import numpy as np
from elemental_factory import ElementalFactory

class CorrelatorFactory(ElementalFactory):
    """
    Thin wrapper around ElementalFactory
        - Load contraction parameters
        - Set cfg_id
        - Load perambulators + elementals
    """

    def __init__(self, ens: str, cfg_id: int):
        # Get contraction parameters first
        tmp = ElementalFactory(ens=ens,
                               cfg_id=cfg_id,
                               nvecs=0,
                               lt=0,
                               ntsrc=0)

        params = tmp.get_contraction_params()

        super().__init__(
            ens=ens,
            cfg_id=cfg_id,
            nvecs=params["nvecs"],
            lt=params["lt"],
            ntsrc=params["ntsrc"],
            tsrc_step=params["tsrc_step"],
        )

        self.flavor_contents = params["flavor_contents"]

    # Public loader
    def load(self, system_name: str = "single"):
        self.load_for_system(system_name)

    # Perambulator passthrough
    # dont think i need this 
    def perambulators(self):
        return super().perambulators()