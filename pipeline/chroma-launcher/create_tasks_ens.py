import argparse
import os
import yaml
import jinja2


FDIR = os.path.dirname(os.path.realpath(__file__))
TEMPLATE_DIR = os.path.join(FDIR, "templates")


# ==========================================================
# YAML LOADING + FLATTENING
# ==========================================================

def load_and_flatten_yaml(yaml_file):
    """
    Load nested YAML and flatten into a single dict
    for Jinja compatibility while keeping structure clean.
    """
    with open(yaml_file) as f:
        raw = yaml.safe_load(f)

    required_sections = ["paths", "cluster", "ensemble", "configs"]
    for section in required_sections:
        if section not in raw:
            raise ValueError(f"Missing required section '{section}' in {yaml_file}")

    flat = {}

    # Flatten sections
    for section in raw:
        if isinstance(raw[section], dict):
            for key, value in raw[section].items():
                flat[key] = value
        else:
            flat[section] = raw[section]

    return flat


# ==========================================================
# TEMPLATE HANDLER
# ==========================================================

class TaskHandler:
    def __init__(self, env):
        self.templates = {
            "eigs": env.get_template("eigs.jinja.xml"),
            "meson": env.get_template("meson.jinja.xml"),
            "meson2": env.get_template("meson2.jinja.xml"),
            "disco": env.get_template("disco.jinja.xml"),
            "peram": env.get_template("peram.jinja.xml"),
            "chroma_eigs": env.get_template("eigs.sh.j2"),
            "chroma_meson": env.get_template("meson.sh.j2"),
            "chroma_meson2": env.get_template("meson2.sh.j2"),
            "chroma_peram": env.get_template("peram.sh.j2"),
            "chroma_disco": env.get_template("disco.sh.j2"),
        }


# ==========================================================
# TASK PROCESSING
# ==========================================================

def determine_task_objects(task_list, data):
    run_objects = []

    for task in task_list:
        if task == "eigs":
            run_objects.extend(["eigs", "chroma_eigs"])

        elif task.startswith("peram"):
            parts = task.split("_")
            if len(parts) != 3:
                raise ValueError("Use peram_<inverter>_<flavor>")

            _, inverter_type, flavor = parts

            data["inverter_type"] = inverter_type
            data["flavor"] = flavor.lower()

            run_objects.extend(["peram", "chroma_peram"])

        elif task == "meson":
            run_objects.extend(["meson", "chroma_meson"])

        elif task == "meson2":
            run_objects.extend(["meson2", "chroma_meson2"])

        elif task == "disco":
            run_objects.extend(["disco", "chroma_disco"])

        else:
            raise ValueError(f"Unknown task: {task}")

    return list(dict.fromkeys(run_objects))


# ==========================================================
# MAIN PROCESSING
# ==========================================================

def process_yaml_file(yaml_file, options, env, handler):
    ens_short = os.path.splitext(os.path.basename(yaml_file))[0]
    print(f"\nProcessing ensemble: {ens_short}")

    data = load_and_flatten_yaml(yaml_file)

    required_keys = [
        "launch_root",
        "cfg_i",
        "cfg_f",
        "cfg_d",
        "NL",
        "NT",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in {yaml_file}")

    run_objects = determine_task_objects(options.list_tasks, data)

    cfg_i = data["cfg_i"]
    cfg_f = data["cfg_f"]
    cfg_d = data["cfg_d"]

    for cfg_id in range(cfg_i, cfg_f, cfg_d):
        print(f"  Generating cfg {cfg_id}")

        for obj in run_objects:

            if obj in ["eigs", "chroma_eigs"]:
                task_dir = "ini-eigs"
            elif obj in ["meson", "chroma_meson"]:
                task_dir = "ini-meson"
            elif obj in ["meson2", "chroma_meson2"]:
                task_dir = "ini-meson2"
            elif obj in ["disco", "chroma_disco"]:
                task_dir = "ini-disco"
            elif obj in ["peram", "chroma_peram"]:
                task_dir = f"ini-perams-{data['flavor']}-{data['inverter_type']}"
            else:
                task_dir = "ini-other"

            output_dir = os.path.join(
                data["launch_root"],
                task_dir,
                f"cnfg{cfg_id:04d}"
            )

            os.makedirs(output_dir, exist_ok=True)

            if obj.startswith("chroma"):
                filename = f"{obj.split('_')[1]}_cfg{cfg_id:04d}.sh"
            else:
                filename = f"{obj}_cfg{cfg_id:04d}.ini.xml"

            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path) and not options.overwrite:
                continue

            render_data = data.copy()
            render_data["cfg_id"] = f"{cfg_id:04d}"
            render_data["ens_short"] = ens_short

            output = handler.templates[obj].render(render_data)

            with open(output_path, "w") as f:
                f.write(output)


# ==========================================================
# ENTRY POINT
# ==========================================================

def main(options):

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        undefined=jinja2.StrictUndefined,
    )

    handler = TaskHandler(env)

    if options.ini_dir:
        for root, _, files in os.walk(options.ini_dir):
            for file in files:
                if file.endswith((".yml", ".yaml")):
                    process_yaml_file(
                        os.path.join(root, file),
                        options,
                        env,
                        handler,
                    )

    elif options.ini:
        process_yaml_file(options.ini, options, env, handler)

    else:
        raise ValueError("Provide --ini or --ini_dir")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ini", type=str)
    parser.add_argument("--ini_dir", type=str)

    parser.add_argument(
        "-l",
        "--list_tasks",
        nargs="+",
        required=True,
    )

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)