#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
from datetime import datetime

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
    # Ensure the yaml path is absolute so SLURM batch scripts can find it
    yaml_path = Path(yaml_file).resolve()
    
    with open(yaml_path) as f:
        full_config = yaml.safe_load(f)

    # Top-level key is ensemble name
    ensemble_key = list(full_config.keys())[0]
    ensemble_cfg = full_config[ensemble_key]
    ensemble_short = ensemble_cfg["ensemble"]["short"]

    # Extract configs
    slurm = ensemble_cfg["slurm"]
    cfg_range = ensemble_cfg["configs"]["range"]
    exclude = set(ensemble_cfg["configs"].get("exclude", []))
    
    paths = ensemble_cfg["paths"]
    base_path = paths["base_path"]
    driver_path = paths["driver_path"]
    venv_activate = paths.get("venv_activate", "")

    # Create run directory using YAML base_path
    run_dir = make_run_dir(base_path)
    corr_dir = run_dir / "correlators"
    log_dir = run_dir / "logs"
    script_dir = run_dir / "scripts"

    # Build configuration IDs list
    all_cfgs = list(range(cfg_range["start"], cfg_range["end"] + 1, cfg_range["step"]))
    valid_cfgs = [c for c in all_cfgs if c not in exclude]

    chunks = list(chunk_list(valid_cfgs, chunk_size))

    # ==============================================================
    # 1. Generate Contraction Batch Scripts
    # ==============================================================
    for i, chunk in enumerate(chunks):
        script_path = script_dir / f"batch_{i:03d}.sh"
        
        with open(script_path, "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={slurm.get('job_name', 'mesonspec')}_{i:03d}\n")
            f.write(f"#SBATCH --account={slurm.get('account', 'exncmf')}\n")
            f.write(f"#SBATCH --nodes={slurm.get('nodes', 1)}\n")
            f.write(f"#SBATCH --ntasks-per-node={slurm.get('ntasks_per_node', 1)}\n")
            f.write(f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 1)}\n")
            f.write(f"#SBATCH --partition={slurm.get('partition', 'dc-cpu')}\n")
            f.write(f"#SBATCH --time={slurm.get('time_minutes', 600)}\n")
            f.write(f"#SBATCH --output={log_dir}/slurm_{i:03d}_%j.out\n")
            f.write(f"#SBATCH --error={log_dir}/slurm_{i:03d}_%j.err\n\n")

            if venv_activate:
                f.write(f"source {venv_activate}\n\n")
            
            f.write(f"YAML_FILE=\"{yaml_path}\"\n")
            f.write(f"CORR_DIR=\"{corr_dir}\"\n")
            f.write(f"SRC_PATH=\"{driver_path}\"\n\n")

            f.write("echo \"Running chunk: " + " ".join(map(str, chunk)) + "\"\n\n")
            
            f.write(f"for CFG_ID in {' '.join(map(str, chunk))}; do\n")
            f.write(f"    PADDED_CFG=$(printf \"%04d\" $CFG_ID)\n")
            f.write(f"    OUTFILE=\"$CORR_DIR/{ensemble_short}_cfg${{PADDED_CFG}}.h5\"\n")
            f.write(f"    if [[ -f \"$OUTFILE\" ]]; then\n")
            f.write(f"        echo \"Skipping cfg $CFG_ID — output already exists\"\n")
            f.write(f"        continue\n")
            f.write(f"    fi\n")
            f.write(f"    srun python3 -u $SRC_PATH --yaml_file \"$YAML_FILE\" --cfg_id $CFG_ID --outdir \"$CORR_DIR\" \\\n")
            f.write(f"         > {log_dir}/cfg_${{PADDED_CFG}}.log 2>&1 &\n")
            f.write(f"done\n\nwait\n")
            
        script_path.chmod(0o755)

    # ==============================================================
    # 2. Generate Post-Processing Script
    # ==============================================================
    # Resolve correct paths for post-processing
    this_script_dir = Path(__file__).resolve().parent
    contractions_dir = this_script_dir.parent
    cij_script = contractions_dir / "postprocess" / "stage2-matrix-assembly" / "single_meson_cij.py"
    
    stage2_outdir = run_dir / "stage2_cij"
    
    post_script_path = script_dir / "postprocess.sh"
    with open(post_script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=post_{ensemble_short}
#SBATCH --account={slurm.get('account', 'exncmf')}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition={slurm.get('partition', 'dc-cpu')}
#SBATCH --time=120
#SBATCH --output={log_dir}/postprocess_%j.out
#SBATCH --error={log_dir}/postprocess_%j.err

if [ -n "{venv_activate}" ]; then
    source {venv_activate}
fi

mkdir -p {stage2_outdir}

echo "Running single_meson_cij..."
python3 -u {cij_script} \\
    --stage1_dir {corr_dir} \\
    --outdir {stage2_outdir}

echo "Post-processing complete."
""")
    post_script_path.chmod(0o755)

    # ==============================================================
    # 3. Generate Master Submission Script
    # ==============================================================
    submit_script_path = run_dir / "submit_all.sh"
    with open(submit_script_path, "w") as f:
        f.write(f"""#!/bin/bash

echo "Submitting contraction jobs..."
JOB_IDS=""

for script in {script_dir}/batch_*.sh; do
    JOB_ID=$(sbatch --parsable $script)
    echo "Submitted $script with Job ID: $JOB_ID"
    
    if [ -z "$JOB_IDS" ]; then
        JOB_IDS="$JOB_ID"
    else
        JOB_IDS="$JOB_IDS:$JOB_ID"
    fi
done

echo ""
echo "Submitting post-processing job with dependency on contractions..."
sbatch --dependency=afterok:$JOB_IDS {post_script_path}
""")
    submit_script_path.chmod(0o755)

    print(f"\nCreated run directory: {run_dir}")
    print(f"Generated {len(chunks)} contraction batch scripts.")
    print(f"Generated post-processing script: {post_script_path}")
    print(f"Master submission script ready. Run the following to start pipeline:")
    print(f" -> bash {submit_script_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM arrays/scripts for single-meson contractions.")
    parser.add_argument("--yaml", required=True, help="Path to the ensemble YAML configuration file.")
    parser.add_argument("--chunk_size", type=int, default=12, help="Number of configurations per SLURM job.")
    args = parser.parse_args()

    generate_batch_scripts(args.yaml, args.chunk_size)