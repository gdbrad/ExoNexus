import numpy as np
from elemental_factory import ElementalFactory


class CorrelatorFactory(ElementalFactory):
    """
    Thin wrapper:
        - Reads ensemble config
        - Sets cfg_id
        - Loads perams + elementals
    """

    def __init__(self, ens: str, cfg_id: int):
        # Let parent load YAML directly
        super().__init__(ens=ens, cfg_id=cfg_id)
        self.flavor_contents = self.io.flavor_contents

    def load(self, system_name: str = "single"):
        self.load_for_system(system_name)