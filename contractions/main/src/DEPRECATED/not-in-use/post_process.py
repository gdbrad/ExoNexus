
import sys
sys.path.append('../')
sys.path.append('../../')
import h5py
import numpy as np
import matplotlib.pyplot as plt

from dimeson_factory import DiMesonOperator
"""after merging h5 files into a single h5 containing all configs, run this postprocessing routine """

#file='contractions/b3.6_s64t64_pi_cfg8000_2pt_nvec_64_tsrc_8_test.h5'
def load_data_tsrc(file,tsrc_avg: bool,Lt:int):
    """
    Load data for all operators and optionally average over 16 tsrc locations.
    The rolling of data arrays is done according to the relation:

    delta_t = t- t_src
    if delta_t <  0:
        delta_t = delta_t + Lt
    
    Args:
        tsrc_avg (bool): If True, averages data over 16 tsrc locations.

    Returns:
        dict: keys are operator names and values are data arrays shape (num_cfgs, Lt, num_tsrcs) or (num_cfgs, Lt) if tsrc_avg=True.
    """
    ops = DiMesonOperator.generate_operators()
    canonical = list(ops.keys())
    n = len(canonical)
    Corr_matrix = np.zeros((n,n,Lt))
    
    with h5py.File(file,'r') as f:
        block = f["Ptot_000_a1p"]
        keys = list(block.keys())
        print(f" integer indexed operator pairs : {len(keys)}")

        loaded = 0 
        for key in keys: 
            if "_X_" not in key: 
                continue 
            parsed = key.split("_X_")

        ops_list = list(f.keys())
        #print(ops_list)

        for ops in ops_list:
            tsrc_groups = sorted([tsrc for tsrc in f[f'{ops}/meson1_light_light'].keys()], key=lambda x: int(x.split('_')[-1]))  
            #tsrc_groups = tsrc_groups[:-1]
            print(tsrc_groups)
            for i,tsrc in enumerate(tsrc_groups):
                cfg_groups = f[f'{ops}/meson1_light_light/{tsrc}'].keys()
                #print(cfg_groups)
                
                # for cfg in cfg_groups:
                #     dataset_names = list(f[f'{ops}/meson1_light_light/{tsrc}/{cfg}'].keys())
                    
                for dataset in cfg_groups:
                    dataset_path = f'{ops}/meson1_light_light/{tsrc}/{dataset}'
                    try:
                        data_array = np.array(f[dataset_path], dtype=np.float64)
                        print('array',data_array)
                        print('len',len(data_array))
                        #data_array[-1] = 0
                        #print(data_array)
                        if (ops, int(tsrc.split('_')[-1])) not in data:
                            data[(ops, int(tsrc.split('_')[-1]))] = []
                        data[(ops, int(tsrc.split('_')[-1]))].append(data_array)
                    except ValueError:
                        print(f"Skipping dataset {dataset_path} due to conversion error")
            # for i,tsrc in enumerate(tsrc_groups):
            #     _tsrc = []
            #     _tsrc.append(tsrc)

                
                if (ops, int(tsrc.split('_')[-1])) in data:
                    data[(ops, int(tsrc.split('_')[-1]))] = np.stack(data[(ops, int(tsrc.split('_')[-1]))])
                    data[(ops, int(tsrc.split('_')[-1]))] = np.roll(data[(ops, int(tsrc.split('_')[-1]))], -int(tsrc.split('_')[-1]) * 8, axis=1)
    
    if tsrc_avg:
        avg_data = {}
        for ops in ops_list:
            valid_tsrcs = [t for _, t in data.keys() if ops in _]
            if valid_tsrcs:
                stacked_data = np.stack([data[(ops, t)] for t in valid_tsrcs], axis=2)  # shape: (ncfg, Lt, 
                #num_tsrcs)
                print('stacked',stacked_data.shape)
                #for i,_ in enumerate(valid_tsrcs):

                    # stacked_data = np.roll([data[(ops, t)] for t in valid_tsrcs],-int(tsrc.split('_')[-1]) * i, axis=2)
                avg_data[ops] = stacked_data.mean(axis=2)  # shape: (ncfg, Lt)
        return avg_data
    
    return data
#file = 'pi_cfg1750_2pt_nvec_64_tsrc_16_test.h5'

def load_data_avg(avg=True):
    data = np.zeros((64, 8))

    with h5py.File(file,'r') as f:
        ops_list = list(f.keys())

        for ops in ops_list:
            tsrc_groups = sorted([tsrc for tsrc in f[f'{ops}/meson1_light_light'].keys()], key=lambda x: int(x.split('_')[-1]))  
            
            for i,tsrc in enumerate(tsrc_groups):
                cfg_groups = f[f'{ops}/meson1_light_light/{tsrc}'].keys()
                t = int(tsrc.split('_')[-1])
                    
                for dataset in cfg_groups:
                    dataset_path = f'{ops}/meson1_light_light/{tsrc}/{dataset}'
                    data_array = np.array(f[dataset_path][:], dtype=np.float64)
                    data[:,i] = data_array
                    print(data.shape)

    if avg:
        for _,tsrc in enumerate(tsrc_groups):
            data[:,_] = np.roll(data[:, _], -int(tsrc.split('_')[-1]) * tsrc, axis=1)
        data = data.mean(axis=0)
    
    return data


#new_data = load_data_avg(avg=True)
new_data = load_data_tsrc(tsrc_avg=True)
print(new_data)
#new_data.keys()
#print(new_data.keys())
new_data = new_data[('pi_000')]
#print('new',new_data)

# pruned_data = np.array(new_data[:-1])
# print(pruned_data)

def fold_data(arr, axis=1):
    T = arr.shape[axis]
    if T % 2 == 1:
        raise ValueError('temporal extent should be even')

    folded_avg = (np.array(arr) + np.flip(arr, axis=axis))/2
    return np.take(folded_avg, range(int(T/2)), axis=axis)

#new_data = load_data_tsrc(tsrc_avg=True)
#_new_data = np.array(new_data['pi_000'][0])
#print(_new_data)
#folded_data = fold_data(new_data)

# print("tr",tr)
# print(data.shape)
    
# data_fold = 0.5*(data + np.roll(tr,0))
# #print(data)
import matplotlib.pyplot
plt.plot(np.arange(64), new_data[0], '.', )

plt.yscale('log')
#plt.legend()
#plt.ylim(-2,20)
plt.savefig('pi_.jpg')

