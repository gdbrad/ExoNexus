#!/usr/bin/env python3
import argparse
import jinja2
import yaml
from pathlib import Path
import re

from scripts.yml_to_xml import eigs_xml, perams_xml, meson_xml, chroma_sh_xml, disco_xml


def parse_ensemble(short_tag: str) -> dict:
    long_tag = {
        "a065m420": "b3.70_ms-0.000_mud-0.020_s32t96-000",
        "a065m300": "b3.70_ms0.000_mud-0.025_s40t96-000",
        "a085m420": "b3.57_ms-0.007_mud-0.038_s24t64-000",
        "a085m300": "b3.57_ms-0.007_mud-0.044_s32t64-000",
        "a085m200": "b3.57_ms-0.007_mud-0.048_s48t64-000",
        "a125m400": "b3.30_ms-0.057_mud-0.1200_s16t64-000",
        "a125m330": "b3.30_ms-0.057_mud-0.1233_s24t64-000",
        "a125m280": "b3.30_ms-0.057_mud-0.1265_s24t64-000",
    }
    if short_tag not in long_tag:
        raise ValueError(f"Unknown ensemble: {short_tag}")

    key = long_tag[short_tag]
    p_match = re.search(r"-(\d{3})$", key)
    P = p_match.group(1) if p_match else "000"

    m = re.match(
        r"b(?P<beta>[0-9]+\.[0-9]+)_ms(?P<ms>[0-9\.\-]+)_mud-(?P<mud>[0-9\.]+)_s(?P<NL>\d+)t(?P<NT>\d+)-\d+",
        key
    )
    if not m:
        raise ValueError(f"Failed to parse: {key}")

    info = m.groupdict()
    info.update({
        "NL": int(info["NL"]),
        "NT": int(info["NT"]),
        "P": P,
    })
    return info


class TaskHandler:
    def __init__(self, env):
        self.templates = {
            'eigs': env.get_template('eigs.jinja.xml'),
            'peram_mg': env.get_template('peram_multigrid.jinja.xml'),
            'peram_clover': env.get_template('peram_clover.jinja.xml'),
            'peram_strange_clover': env.get_template('peram_strange_clover.jinja.xml'),
            'peram_charm_mg': env.get_template('peram_charm_multigrid.jinja.xml'),
            'peram_strange_mg': env.get_template('peram_strange_multigrid.jinja.xml'),
            'meson': env.get_template('meson.jinja.xml'),
            'disco': env.get_template('disco.jinja.xml'),
            'chroma_eigs': env.get_template('eigs.sh.j2'),
            'chroma_peram_mg': env.get_template('peram_jureca.sh.j2'),
            'chroma_peram_clover': env.get_template('peram_clover.sh.j2'),
            'chroma_peram_charm_mg': env.get_template('peram_charm_mg.sh.j2'),
            'chroma_peram_strange_mg': env.get_template('peram_strange_mg.sh.j2'),
            'chroma_peram_strange_clover': env.get_template('peram_strange_clover.sh.j2'),

            'chroma_meson': env.get_template('meson_jureca.sh.j2'),
            'chroma_disco': env.get_template('disco.sh.j2'),
        }


def process_yaml_file(yaml_file: Path, options, env, handler):
    ens_short = yaml_file.stem
    print(f"\nProcessing ensemble: {ens_short}")

    with open(yaml_file) as f:
        dataMap = yaml.safe_load(f)

    paths = dataMap["paths"]
    data_path = Path(paths["data_path"]).expanduser()
    eigs_path = Path(paths["eigs_path"]).expanduser()
    launch_path = Path(paths["launch_path"]).expanduser()
    cfg_path = Path(paths["cfg_path"]).expanduser()
    cfg_name = paths["cfg_name"]  # e.g. "b3.70_ms0.000_mud-0.025_s40t96-000-n_cfg_"

    print(f"Data path   : {data_path}")
    print(f"Launch path : {launch_path}")
    print(f"Config path : {cfg_path}")
    print(f"Config name : {cfg_name}")

    ens_props = parse_ensemble(ens_short)
    dataMap.update(ens_props)
    dataMap['ens_short'] = ens_short
    dataMap['Nt_forward'] = ens_props['NT']
    dataMap['prop_t_fwd'] = ens_props['NT']
    dataMap['meson_t_fwd'] = ens_props['NT']
    dataMap['NL'] = ens_props['NL']

    # 1. SBD directories in data_path
    sdb_dirs = ['eigs_sdb', 'perams_sdb', 'meson_sdb', 'perams_charm_sdb', 'perams_strange_sdb', 'chroma_out']
    for d in sdb_dirs:
        (data_path / d).mkdir(parents=True, exist_ok=True)

    # 2. Launch directories in launch_path
    run_path = launch_path
    run_path.mkdir(parents=True, exist_ok=True)

    task_map = {
        'eigs': ['eigs', 'chroma_eigs'],
        'peram_mg': ['peram_mg', 'chroma_peram_mg'],
        'peram_clover': ['peram_clover', 'chroma_peram_clover'],
        'peram_strange_clover': ['peram_strange_clover', 'chroma_peram_strange_clover'],
        'peram_charm_mg': ['peram_charm_mg', 'chroma_peram_charm_mg'],
        'peram_strange_mg': ['peram_strange_mg', 'chroma_peram_strange_mg'],
        'meson': ['meson', 'chroma_meson'],
        'disco': ['disco', 'chroma_disco'],
    }
    run_objects = []
    for task in options.list_tasks:
        run_objects.extend(task_map.get(task, []))
    run_objects = list(dict.fromkeys(run_objects))

    cfg_i = dataMap['cfg_i']
    cfg_f = dataMap['cfg_f']
    cfg_step = dataMap['cfg_d']

    written = 0
    for cfg_id in range(cfg_i, cfg_f + 1, cfg_step):
        cfg_str = f"{cfg_id}"
        print(f"  → cfg {cfg_str}")

        # Full gauge config filename
        gauge_file = cfg_path / f"{cfg_name}{cfg_str}.lime"

        for obj in run_objects:
            subdir_map = {
                'eigs': 'ini-eigs', 'chroma_eigs': 'ini-eigs',
                'peram_clover': 'ini-perams-clover', 'chroma_peram_clover': 'ini-perams-clover',
                'peram_strange_clover': 'ini-perams-strange', 'chroma_peram_strange_clover': 'ini-perams-strange',
                'peram_mg': 'ini-perams', 'chroma_peram_mg': 'ini-perams',
                'peram_charm_mg': 'ini-perams-charm', 'chroma_peram_charm_mg': 'ini-perams-charm',
                'peram_strange_mg': 'ini-perams-strange', 'chroma_peram_strange_mg': 'ini-perams-strange',
                'meson': 'ini-meson', 'chroma_meson': 'ini-meson',
                'disco': 'ini-disco', 'chroma_disco': 'ini-disco',
            }
            subdir = subdir_map.get(obj, "ini-other")
            obj_dir = run_path / subdir / f"cnfg{cfg_str}"
            obj_dir.mkdir(parents=True, exist_ok=True)

            if obj.startswith("chroma_"):
                name = obj.split("_", 1)[1]
                ext = ".sh"
            else:
                name = obj
                ext = ".ini.xml"

            out_file = obj_dir / f"{name}_cfg{cfg_str}{ext}"

            if out_file.exists() and not options.overwrite:
                continue

            render_data = dataMap.copy()
            render_data['cfg_id'] = cfg_str
            render_data['data_path'] = str(data_path)
            render_data['cfg_path'] = str(cfg_path)
            render_data['eigs_path'] = str(eigs_path)

            render_data['launch_path'] = str(launch_path)
            render_data['gauge_file'] = str(gauge_file)
            render_data['momentum_list'] = meson_xml._gen_mom_list()
            render_data['displacement_list'] = meson_xml._displacement_list()

            try:
                content = handler.templates[obj].render(render_data)
                with open(out_file, "w") as f:
                    f.write(content)
                print(f"    [OK] {out_file.name}")
                written += 1
            except Exception as e:
                print(f"    [ERROR] {obj}: {e}")

    print(f"\nFinished: {written} files written in {launch_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ini', type=str)
    parser.add_argument('--ini_dir', type=str)
    parser.add_argument('-l', '--list_tasks', nargs='+', required=True)
    parser.add_argument('--overwrite', action='store_true')
    options = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    template_dir = script_dir / "scripts" / "templates"
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template dir not found: {template_dir}")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        undefined=jinja2.StrictUndefined
    )
    handler = TaskHandler(env)

    if options.ini:
        process_yaml_file(Path(options.ini), options, env, handler)
    elif options.ini_dir:
        for f in Path(options.ini_dir).rglob("*.yml"):
            process_yaml_file(f, options, env, handler)
    else:
        parser.error("Need --ini or --ini_dir")


if __name__ == "__main__":
    main()
