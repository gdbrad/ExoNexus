'''just recycling other chroma input file generation script to generate binned srun tasks (serial) for perams and mesons in groups of 4'''
import argparse
import os
import jinja2
import yaml
from yml_to_xml import perams_xml, meson_xml, chroma_sh_xml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FDIR = os.path.dirname(os.path.realpath(__file__))
TEMPLATE = os.path.join(FDIR, 'templates')
RESULTPATH = os.path.join(FDIR, 'res')
OUTPATH = os.path.join(RESULTPATH, 'out')
LOGPATH = os.path.join(RESULTPATH, 'log')

class TaskHandler:
    def __init__(self, env):
        self.templates = { 
            'chroma_peram': env.get_template('peram_mg_binned.sh.j2'),
            'chroma_peram_strange': env.get_template('peram_strange_mg_binned.sh.j2'),
            'chroma_peram_charm': env.get_template('peram_charm_mg_binned.sh.j2'),
            'chroma_meson': env.get_template('meson_binned.sh.j2'),
        }
        self.xml_classes = {
            'chroma_meson': chroma_sh_xml.ChromaOptions,
            'chroma_peram': chroma_sh_xml.ChromaOptions,
            'chroma_peram_strange': chroma_sh_xml.ChromaOptions,
            'chroma_peram_charm': chroma_sh_xml.ChromaOptions,

        }

def main(options):
    with open(os.path.join(options.in_file)) as f:
        dataMap = yaml.safe_load(f)
    missing_values = [key for key in ['run_path', 'cfg_path'] if key not in dataMap]

    if missing_values:
        for key in missing_values:
            value = input(f"you forgot to include '{key}' in your infile dummy!: ")
            dataMap[key] = value

        with open(options.in_file, 'w') as f:
            yaml.safe_dump(dataMap, f)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE), undefined=jinja2.StrictUndefined)
    handler = TaskHandler(env)
    run_objects = []
    for task in options.list_tasks:
        if 'peram' in task:
            run_objects.append('chroma_peram')
        if 'peram_strange' in task:
            run_objects.append('chroma_peram_strange')
        if 'peram_charm' in task:
            run_objects.append('chroma_peram_charm')
        if 'meson' in task:
            run_objects.append('chroma_meson')
    cfg_step=50
    group_size=4 # change this number of in files per slurm script
    devices = [0, 1, 2, 3]  # Assuming 4 GPUs are available

    for cfg_group_start in range(options.cfg_i, options.cfg_f, cfg_step * group_size):
        cfg_group_end = min(cfg_group_start + cfg_step *(group_size-1), options.cfg_f)
        cfg_ids = range(cfg_group_start, cfg_group_end + cfg_step, cfg_step)
        cfg_group_name = f'cfgs_{cfg_group_start}-{cfg_group_end}'
        print(f'Creating scripts and directories for configuration group {cfg_group_name}')
        
        group_dir = os.path.join(options.run_dir, cfg_group_name)
        print(os.path.realpath(group_dir))
        os.makedirs(group_dir, exist_ok=True)

        for nvec in options.num_vecs:
            nvec_dir = os.path.join(group_dir, f'numvec{nvec}')
            # print(os.path.realpath(nvec_dir))
            os.makedirs(nvec_dir, exist_ok=True)

            for obj in run_objects:
                if obj.startswith('chroma_'):
                    _obj = obj[len('chroma_'):]
                    ini_out = f'{_obj}_{nvec}_cfgs_{cfg_group_start}-{cfg_group_end}.sh'
                    ini_path = os.path.join(nvec_dir,ini_out)
                    cfg_ids_and_devices = list(zip(cfg_ids, devices))
                    # print(ini_path)
                else:
                    for cfg_id in cfg_ids:
                        ini_out = f'{obj}_{nvec}_cfg_{cfg_id}.ini.xml'
                        ini_path = os.path.join(nvec_dir, ini_out)
                        print(ini_path)
                

                with open(options.in_file) as f:
                    dataMap = yaml.safe_load(f)

                    if obj in options.list_tasks:
                        base = handler.xml_classes[obj]
                        # if dataMap['facility'] in ['jureca', 'juwels']:
                        #     expected_keys = base.__fields__.keys()
                        # else:
                        expected_keys = base.model_fields.keys()
                        filtered_data = {k: v for k, v in dataMap.items() if k in expected_keys}
                        # filtered_data['cfg_id'] = f'{cfg_id:02d}'
                        filtered_data['cfg_ids'] = cfg_ids
                        filtered_data['cfg_ids_and_devices'] = cfg_ids_and_devices
                        filtered_data['cfg_group_start'] = cfg_group_start
                        filtered_data['cfg_group_end'] = cfg_group_end

                        moms = meson_xml._gen_mom_list()
                        disp = meson_xml._displacement_list()
                        filtered_data['momentum_list'] = moms
                        filtered_data['displacement_list'] = disp
                        # default value in yml file is overridden by the nvec loop for nvec study
                        filtered_data['num_vecs_perams'] = nvec
                        filtered_data['meson_nvec'] = nvec
                        ens_props = chroma_sh_xml.parse_ensemble(short_tag=filtered_data['ens_short'])
                        filtered_data.update(ens_props)
                        output_xml = handler.templates[obj].render(filtered_data)
                        
                        print(f"Writing file {ini_path} for object {obj}")
                        
                        with open(ini_path, 'w') as f:
                            f.write(output_xml)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--cfg_i', type=int, required=True)
    parser.add_argument('--cfg_f', type=int, required=True)
    parser.add_argument('--cfg_step', type=int, nargs='?', default=10, help='default: %(default)s')
    parser.add_argument('-l', '--list_tasks', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-nv', '--num_vecs', nargs='+', help='<Required> number of eigenvectors to use for perams and mesons, must match', required=True, type=int)
    parser.add_argument('--overwrite', type=bool, nargs='?', default=False, help='if true, overwrite existing xml and sh scripts for chroma task of interest')
    parser.add_argument('--run_dir', default='', help='default: %(default)s')
    options = parser.parse_args()
    main(options)