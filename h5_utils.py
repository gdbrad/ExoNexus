import h5py as h5
# import gamma
import numpy as np 
from typing import Dict
from typing import Optional
from typing import Union
from typing import List
from typing import Any 
import os
# import pprint

def move_groups_to_parent(file):
    with h5.File(file,'a') as f:
        parent_name = 'a1_mp'
        if parent_name not in f:
            parent_grp = f.create_group(parent_name)
        operator_groups = [
        'b_1xNABLA_A1',
        'pion',
        'pion_2',
        'rho_2xB_A1',
        'rhoxB_A1'
    ]
        
        for op_group in operator_groups:
            if op_group in f:
                f.move(op_group, f"{parent_name}/{op_group}")
                print(f"Moved group '{op_group}' to '{parent_name}/{op_group}'.")
            else:
                print(f"Group '{op_group}' does not exist, skipping.")

def create_dset(h5f: str, key: str, data: Any, overwrite: bool = False):
    """Creates or overwrites (if requested) dataset in HDF5 file.

    **Arguments**
        h5f: h5py.File
            The file to write to.

        key: str
            The name of the dataset.

        data: Any
            The data for the dataset.

        overwrite: bool = False
            Wether data shall be overwritten.
    """
    # LOGGER.debug("Writing dataset:`%s`", key)
    if key in h5f:
        if overwrite:
            del h5f[key]
            h5f.create_dataset(key, data=data)
        else:
            print("Skipping dataset because exists:`%s`", key)
    else:
        f = h5.File(h5f,"w")
        f.create_dataset(key, data=data)
def get_dsets(
    container: Union[h5.File, h5.Group],
    parent_name: Optional[str] = None,
    load_dsets: bool = False,
    ignore_containers: Optional[List[str]] = None,
) -> Dict[str, Union[h5.Dataset, np.ndarray]]:
    """Access an HDF5 container and extracts datasets.

    The method is iteratively called if the container contains further containers.

    **Arguments**
        container: Union[h5py.File, h5py.Group]
            The HDF5 group or file to iteratetively search.

        parent_name: Optional[str] = None
            The name of the parent container.

        load_dsets: bool = False
            If False, data sets are not opened (lazy load).
            If True, returns Dict with numpy arrays as values.

        ignore_containers: Optional[List[str]] = None
            A list of HDF5 containers to ignore when iteratively solving.
            Can be regex expressions.

    **Returns**
        datasets: Dict[str, Union[h5py.Dataset, np.ndarray]]
            A dictionary containing the full path HDF path (e.g., `groupA/subgroupB`)
            to the data set as keys and the unloaded values of the set as values.
    """

    dsets = {}
    ignore_containers = [] if ignore_containers is None else ignore_containers
    for key in container:
        obj = container[key]

        address = os.path.join(parent_name, key) if parent_name else key

        if isinstance(obj, h5.Dataset):
            dsets[address] = obj[()] if load_dsets else obj
        elif isinstance(obj, h5.Group):
            dsets.update(get_dsets(obj, parent_name=address, load_dsets=load_dsets))

    return dsets