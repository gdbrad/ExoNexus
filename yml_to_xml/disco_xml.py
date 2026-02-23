from pydantic import BaseModel, Field
from typing import List

def _gen_mom_list():
   momentum_list: List[str] = [
        "0 0 0", "1 0 0", "-1 0 0",
        "0 1 0", "0 -1 0", "0 0 1",
        "0 0 -1", "2 0 0", "-2 0 0",
        "0 2 0", "0 -2 0", "0 0 2",
        "0 0 -2", "3 0 0", "-3 0 0",
        "0 3 0", "0 -3 0", "0 0 3",
        "0 0 -3"
    ]
   return momentum_list


def  _displacement_list()->List[str]: 
    displacement_list= ['', '1', '2', '3', '1 1', '2 2', '3 3', '1 2', '1 3', '2 1', '2 3', '3 1', '3 2']
    return displacement_list

class Disco(BaseModel):
    '''Disco dataclass matching that of input yaml files 
    Output -> xml file 
    '''

    # universal imports for a given ensemble 
    NL: int
    NT: int
    prop_mass_light_label: str 
    prop_clov_coeff: int 
    cfg_name: str 

    t_start: int
    ens_short: str
    disco_t_source_list: list #HARDCODED RIGHT NOW 0-64
    disco_max_rhs: int
    disco_probing_displacement: int
    disco_probing_power: int
    disco_max_colors_at_once: int
    disco_max_colors: int
    disco_noise_vectors: int
    num_color_parts:int = Field(init=False)
    disco_displacement_list: List[str] = ['', '1', '2', '3', '1 1', '2 2', '3 3', '1 2', '1 3', '2 1', '2 3', '3 1', '3 2']

    run_path:str
    #link smearing options 
    cfg_path: str
    num_color_parts: int = 14

    def __post_init__(self):
        # Calculate num_color_parts after initialization
            self.num_color_parts = (self.disco_max_colors + self.disco_max_colors_at_once - 1) // self.disco_max_colors_at_once

            3325 + 256-1 // 256