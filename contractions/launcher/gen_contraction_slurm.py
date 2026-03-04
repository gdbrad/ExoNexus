import yaml
from pathlib import Path
from datetime import datetime
import argparse
from itertools import count

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

from pathlib import Path
from datetime import datetime

def make_run_dir(ensemble_short: str) -> Path:
    base = Path("/p/project1/exflash/contractions-tmp")
    date_str = datetime.now().strftime("%Y%m%d")

    ens_dir = base / ensemble_short
    ens_dir.mkdir(parents=True, exist_ok=True)

    run_dir = ens_dir / f"run-{date_str}"
    attempt = 1
    while run_dir.exists():
        run_dir = ens_dir / f"run-{date_str}_{attempt}"
        attempt += 1

    run_dir.mkdir()

    # Create subdirs
    (run_dir / "correlators").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "scripts").mkdir()

    return run_dir

def generate_batch_scripts(yaml_file: str, tmp_base: str = "./tmp", chunk_size: int = 12):
    yaml_path = Path(yaml_file)
    with open(yaml_file) as f:
        full_config = yaml.safe_load(f)

    # Determine ensemble name (from ensemble.short)
    if "ensemble" in full_config and "short" in full_config["ensemble"]:
        ensemble_name = full_config["ensemble"]["short"]
    else:
        raise ValueError("YAML file must contain ensemble.short")

    # Now the actual config dict for batch generation
    cfg = full_config

    # slurm, configs, etc.
    slurm = cfg["slurm"]
    cfgs_cfg = cfg["configs"]
    # Build list of cfg_ids
    if "range" in cfgs_cfg and cfgs_cfg["range"] is not None:
        r = cfgs_cfg["range"]
        cfg_ids = list(range(r["start"], r["end"] + r.get("step", 1), r.get("step", 1)))
    elif "list" in cfgs_cfg and cfgs_cfg["list"] is not None:
        cfg_ids = cfgs_cfg["list"]
    else:
        raise ValueError("Need range or list")

    exclude = set(cfgs_cfg.get("exclude", []))
    cfg_ids = [c for c in cfg_ids if c not in exclude]

    ensemble_name = full_config["ensemble"]["short"]
    run_dir = make_run_dir(ensemble_name)
    corr_dir = run_dir / "correlators"
    log_dir = run_dir / "logs"
    script_dir = run_dir / "scripts"
    driver_path = cfg["paths"]["driver_path"]

    # Time format
    time_val = slurm["time_minutes"]
    if isinstance(time_val, (int, str)):
        hours, minutes = divmod(int(time_val), 60)
        time_val = f"{hours:02d}:{minutes:02d}:00"

    # Split into chunks
    chunks = list(chunk_list(cfg_ids, chunk_size))

    for idx, chunk in enumerate(chunks, 1):
        #script_path = run_dir / f"contract_{ensemble_name}_part{idx:02d}.sh"
        script_path = script_dir / f"contract_{ensemble_name}_part{idx:02d}.sh"
        with open(script_path, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name={slurm['job_name']}_p{idx:02d}
#SBATCH --account={slurm['account']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --time={time_val}
#SBATCH --partition={slurm['partition']}
#SBATCH --ntasks-per-node={slurm.get('ntasks_per_node', 1)}
#SBATCH --output={log_dir}/%j.out
#SBATCH --error={log_dir}/%j.err

echo "=== Job $SLURM_JOB_ID | Part {idx}/{len(chunks)} | $(date) ==="
echo "Configs: {' '.join(map(str, chunk))}"

SRC_PATH="{driver_path}"
YAML_FILE="{yaml_path}"
CORR_DIR="{corr_dir}"

for CFG_ID in {' '.join(map(str, chunk))}; do
    OUTFILE="{corr_dir}/{ensemble_name}_cfg${{CFG_ID}}.h5"
    if [[ -f "$OUTFILE" ]]; then
        echo "Skipping cfg $CFG_ID — output already exists"
        continue
    fi
    srun python3 -u $SRC_PATH --yaml_file "$YAML_FILE" --cfg_id $CFG_ID --outidr "$CORR_DIR"\\
         > {log_dir}/cfg_${{CFG_ID}}.log 2>&1 &
done

wait
echo "Part {idx} finished — correlators in {corr_dir}"
""")
        script_path.chmod(0o755)
        print(f"[INFO] Generated {script_path} → {len(chunk)} configs")

    print(f"\nDone. Launch with:\n   sbatch {run_dir}/contract_{ensemble_name}_part*.sh")
    print(f"\nFinal correlators will be in:\n   {corr_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--chunk", type=int, default=12)
    args = parser.parse_args()
    generate_batch_scripts(args.yaml_file, args.chunk)