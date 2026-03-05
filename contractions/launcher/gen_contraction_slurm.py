#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from itertools import islice

def chunk_list(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def make_run_dir(base_path: str) -> Path:
    """Create a timestamped run directory with attempt suffix to avoid overwriting."""
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    attempt = 1
    
    while True:
        run_dir = base / f"run-{date_str}_{attempt}"
        if not run_dir.exists():
            break
        attempt += 1

    run_dir.mkdir()
    # Create subdirectories
    (run_dir / "correlators").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "scripts").mkdir()
    return run_dir

def generate_batch_scripts(yaml_file: str, chunk_size: int = 12):
    yaml_path = Path(yaml_file).resolve()
    with open(yaml_path) as f:
        full_config = yaml.safe_load(f)

    # Get first ensemble key
    ensemble_key = list(full_config.keys())[0]
    ensemble_cfg = full_config[ensemble_key]

    # ensemble.short must exist
    if "ensemble" in ensemble_cfg and "short" in ensemble_cfg["ensemble"]:
        ensemble_short = ensemble_cfg["ensemble"]["short"]
    else:
        raise ValueError("YAML file must contain ensemble.short")

    # SLURM settings and paths
    slurm = ensemble_cfg["slurm"]
    cfgs_cfg = ensemble_cfg["configs"]
    driver_path = ensemble_cfg["paths"]["driver_path"]
    base_path = ensemble_cfg["paths"]["base_path"]

    YAML_FILE = str(yaml_path)  # for script formatting

    # Create the run directory using the YAML base_path
    run_dir = make_run_dir(base_path)
    corr_dir = run_dir / "correlators"
    log_dir = run_dir / "logs"
    script_dir = run_dir / "scripts"

    # Build list of configuration IDs
    if "range" in cfgs_cfg and cfgs_cfg["range"] is not None:
        r = cfgs_cfg["range"]
        step = r.get("step", 1)
        cfg_ids = list(range(r["start"], r["end"] + 1, step))
    elif "list" in cfgs_cfg and cfgs_cfg["list"] is not None:
        cfg_ids = cfgs_cfg["list"]
    else:
        raise ValueError("Need range or list in configs")

    exclude = set(cfgs_cfg.get("exclude", []))
    cfg_ids = [c for c in cfg_ids if c not in exclude]

    # Format SLURM time
    time_val = slurm.get("time_minutes", 60)
    if isinstance(time_val, (int, str)):
        hours, minutes = divmod(int(time_val), 60)
        time_val = f"{hours:02d}:{minutes:02d}:00"

    # Split configs into chunks
    chunks = list(chunk_list(cfg_ids, chunk_size))

    # Generate scripts for each chunk
    for idx, chunk in enumerate(chunks, 1):
        script_path = script_dir / f"contract_{ensemble_short}_part{idx:02d}.sh"
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
YAML_FILE="{YAML_FILE}"
CORR_DIR="{corr_dir}"

for CFG_ID in {' '.join(map(str, chunk))}; do
    OUTFILE="{corr_dir}/{ensemble_short}_cfg${{CFG_ID}}.h5"
    if [[ -f "$OUTFILE" ]]; then
        echo "Skipping cfg $CFG_ID — output already exists"
        continue
    fi
    srun python3 -u $SRC_PATH --yaml_file "$YAML_FILE" --cfg_id $CFG_ID --outdir "$CORR_DIR" \\
         > {log_dir}/cfg_${{CFG_ID}}.log 2>&1 &
done

wait
echo "Part {idx} finished — correlators in {corr_dir}"
""")
        script_path.chmod(0o755)
        print(f"[INFO] Generated {script_path} → {len(chunk)} configs")

    print(f"\nDone. Launch with:\n   sbatch {run_dir}/scripts/contract_{ensemble_short}_part*.sh")
    print(f"\nFinal correlators will be in:\n   {corr_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--chunk", type=int, default=12)
    args = parser.parse_args()
    generate_batch_scripts(args.yaml_file, args.chunk)