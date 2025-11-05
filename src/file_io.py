from typing import List
import abc 
import os 
from ingest_data import load_elemental,load_peram,reverse_perambulator_time
from __init__ import MESON_NAME_MAP, DI_MESON_NAME_MAP, BASE_PATH,TAPE_PATH

class FileIO:
    def __init__(self,
                 flavor_contents: List[str],
                 cfg_id: int,
                 ens: str,
                 nvecs: int,
                 LT: int,
                 num_tsrc: int,
                 tsrc_step: int,
                 data1:bool):
        self.flavor_contents = flavor_contents
        self.cfg_id = cfg_id
        self.ens = ens
        self.nvecs = nvecs
        self.LT = LT
        self.num_tsrc = num_tsrc
        self.tsrc_step = tsrc_step
        self.data1 = data1
        # Define directories
        if self.data1:
            self.dirs = {
                'light': os.path.join(TAPE_PATH, ens, 'perams_h5'),
                'meson': os.path.join(TAPE_PATH, ens, 'meson_h5'),
                'strange': os.path.join(BASE_PATH, ens, 'perams_strange_sdb'),
                'charm': os.path.join(BASE_PATH, ens, 'perams_charm_sdb')
            }
        else:
            self.dirs = {
                'light': os.path.join(BASE_PATH, ens, 'perams_sdb', f'numvec{nvecs}'),
                'meson': os.path.join(BASE_PATH, ens, 'meson_sdb'),
                'strange': os.path.join(BASE_PATH, ens, 'perams_strange_sdb'),
                'charm': os.path.join(BASE_PATH, ens, 'perams_charm_sdb')
            }

        # Preload meson elemental and perambulator data
        meson_path = os.path.join(self.dirs['meson'], f"meson-{self.nvecs}_cfg{self.cfg_id}.h5")
        if not os.path.exists(meson_path):
            print(f"Meson elemental file not found: {meson_path}")
            self.meson_elemental = None
        else:
            self.meson_elemental = load_elemental(meson_path, LT, nvecs, mom='mom_0_0_0', disp='disp')
        
        self.peram_light = load_peram(os.path.join(self.dirs['light'], f"peram_{nvecs}_cfg{cfg_id}.h5"), 
                                      LT, nvecs, num_tsrc, tsrc_step) if 'light_light' in flavor_contents or 'light_strange' in flavor_contents or 'light_charm' in flavor_contents else None
        self.peram_charm = load_peram(os.path.join(self.dirs['charm'], f"peram_charm_nv{nvecs}_cfg{cfg_id}.h5"), 
                                      LT, nvecs, num_tsrc, tsrc_step) if 'light_charm' in flavor_contents or 'charm_strange' in flavor_contents or 'charm_charm' in flavor_contents else None
        self.peram_strange = load_peram(os.path.join(self.dirs['strange'], f"peram_strange_nv{nvecs}_cfg{cfg_id}.h5"), 
                                        LT, nvecs, num_tsrc, tsrc_step) if 'light_strange' in flavor_contents or 'charm_strange' in flavor_contents else None

    def peram_data(self):
        peram_data = {}
        flavor_map = self.flavor_map()
        for idx, flavor_content in enumerate(self.flavor_contents, 1):
            if flavor_content not in flavor_map:
                print(f"Flavor '{flavor_content}' not found in flavor_map. Skipping.")
                return False
            peram, peram_back_data, peram_file, peram_back_file = flavor_map[flavor_content]
            if peram_back_data is None:
                print(f"Required {flavor_content} back perambulator file missing: {peram_back_file}. Skipping.")
                return False
            peram_back = reverse_perambulator_time(peram_back_data)
            print(f"Flavor '{flavor_content}' (meson {idx}):")
            print(f"  Perambulator loaded: {peram_file}")
            print(f"  Reverse perambulator loaded: {peram_back_file}")

            # Store perambulator data for di-meson contractions
            peram_data[flavor_content] = (peram, peram_back)

        return peram_data
    
    def file_specs(self):
        """File naming conventions for a single cfg."""
        return {
            'light': (self.dirs['light'], f"peram_{self.nvecs}_cfg{{cfg_id}}.h5"),
            'meson': (self.dirs['meson'], f"meson-{self.nvecs}_cfg{{cfg_id}}.h5"),
            'strange': (self.dirs['strange'], f"peram_strange_nv{self.nvecs}_cfg{{cfg_id}}.h5"),
            'charm': (self.dirs['charm'], f"peram_charm_nv{self.nvecs}_cfg{{cfg_id}}.h5")
        }

    def get_meson_system_name(self) -> str:
        """Return the name of the meson system based on flavor contents."""
        if len(self.flavor_contents) == 1:
            return MESON_NAME_MAP.get(self.flavor_contents[0], self.flavor_contents[0])
        else:
            flavor_pair = tuple(self.flavor_contents)
            return DI_MESON_NAME_MAP.get(flavor_pair, '_'.join(self.flavor_contents))

    def get_file_path(self, flavor: str = 'meson') -> str | None:
        """Construct file path for a given flavor and check if it exists."""
        file_specs = self.file_specs()
        if flavor not in file_specs:
            print(f"Flavor '{flavor}' not recognized in file_specs.")
            return None
        directory, filename = file_specs[flavor]
        full_path = os.path.join(directory, filename.format(cfg_id=self.cfg_id))
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}. Skipping.")
            return None
        return full_path

    def _get_required_flavors(self) -> set:
        """Determine the required quark flavors based on flavor contents."""
        required_flavors = set()
        for flavor_content in self.flavor_contents:
            if flavor_content == 'light_light':
                required_flavors.add('light')
            elif flavor_content == 'light_strange':
                required_flavors.update(['light', 'strange'])
            elif flavor_content == 'light_charm':
                required_flavors.update(['light', 'charm'])
            elif flavor_content == 'charm_strange':
                required_flavors.update(['charm', 'strange'])
            elif flavor_content == 'charm_charm':
                required_flavors.add('charm')
            else:
                print(f"Flavor '{flavor_content}' not recognized, defaulting to light.")
                required_flavors.add('light')
        return required_flavors

    @property
    def get_required_flavors(self) -> set:
        """Property to access required flavors."""
        return self._get_required_flavors()

    def flavor_map(self) -> dict:
        """Construct flavor map based on available perambulators."""
        flavor_map = {}
        required_flavors = self.get_required_flavors
        paths = self.file_specs()
        for flavor_content in self.flavor_contents:
            if flavor_content not in {'light_light', 'light_strange', 'light_charm', 'charm_strange', 'charm_charm'}:
                print(f"Flavor '{flavor_content}' not recognized, defaulting to light_light.")
                flavor_content = 'light_light'

            if flavor_content == 'light_light':
                flavor_map[flavor_content] = (self.peram_light, self.peram_light, paths.get('light'), paths.get('light'))
            elif flavor_content == 'light_strange' and 'strange' in required_flavors:
                flavor_map[flavor_content] = (self.peram_light, self.peram_strange, paths.get('light'), paths.get('strange'))
            elif flavor_content == 'light_charm' and 'charm' in required_flavors:
                flavor_map[flavor_content] = (self.peram_light, self.peram_charm, paths.get('light'), paths.get('charm'))
            elif flavor_content == 'charm_strange' and 'charm' in required_flavors and 'strange' in required_flavors:
                flavor_map[flavor_content] = (self.peram_charm, self.peram_strange, paths.get('charm'), paths.get('strange'))
            elif flavor_content == 'charm_charm':
                flavor_map[flavor_content] = (self.peram_charm, self.peram_charm, paths.get('charm'), paths.get('charm'))
            else:
                print(f"Required perambulators for {flavor_content} not available. Defaulting to light_light.")
                flavor_map[flavor_content] = (self.peram_light, self.peram_light, paths.get('light'), paths.get('light'))

        return flavor_map