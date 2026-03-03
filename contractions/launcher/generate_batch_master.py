import yaml
import os
from datetime import datetime
import argparse
from pathlib import Path

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_batch_scripts(ini_file: str, chunk_size: int = 12):
    # 
    ini_path = Path(ini_file).resolve()
    ini_dir = ini_path.parent

    with open(ini_file, 'r') as f:
        full_config = yaml.safe_load(f)

    ens = list(full_config.keys())[0]
    config = full_config[ens]
    slurm = config['slurm']
    cfgs = config['configs']

    # Build full cfg list
    if 'range' in cfgs and cfgs['range'] is not None:
        r = cfgs['range']
        cfg_ids = list(range(r['start'], r['end'] + r.get('step', 1), r.get('step', 1)))
    elif 'list' in cfgs and cfgs['list'] is not None:
        cfg_ids = cfgs['list']
    else:
        raise ValueError("Need range or list")

    exclude = set(cfgs.get('exclude', []))
    print(exclude)
    cfg_ids = [c for c in cfg_ids if c not in exclude]

    # Output directories
    now = datetime.now()
    base_out = Path(slurm.get('output_dir', str(ini_dir))).expanduser()
    run_dir = base_out / f"run-{now.strftime('%Y%m%d')}"
    log_dir = run_dir / "logs"
    corr_dir = run_dir / "correlators"          # ← NEW: where final .h5 files go

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    corr_dir.mkdir(exist_ok=True)

    # Fix time format
    time_val = slurm['time']
    if isinstance(time_val, (int, str)) and ':' not in str(time_val):
        time_val = f"{time_val}:00:00"

    # Pattern of final correlator file (adjust if your filename differs slightly)
    done_pattern = f"{ens}_Dpi_cfg{{cfg}}_matrix_n64_ntsrc8_*.h5"

    # Skip configs that already have their final correlator in THIS run dir
    remaining_cfgs = []
    for cfg in cfg_ids:
        if any((corr_dir / done_pattern.format(cfg=cfg)).glob("*")):
            print(f"Skipping cfg {cfg:4d} — correlator already exists in {corr_dir}")
        else:
            remaining_cfgs.append(cfg)

    if not remaining_cfgs:
        print("All configurations already completed!")
        return

    print(f"{len(remaining_cfgs)} / {len(cfg_ids)} configurations still need to be done")
    chunks = list(chunk_list(remaining_cfgs, chunk_size))

    for idx, chunk in enumerate(chunks, 1):
        script_path = run_dir / f"contract_{ens}_part{idx:02d}.sh"

        with open(script_path, 'w') as f:
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

source /p/scratch/exflash/sc_venv_template/activate.sh

SRC_PATH='/p/project1/exflash/gdb-code/exotraction/src/single_meson_driver.py'
YAML_FILE="{ini_path}"
CORR_DIR="{corr_dir}"

echo "=== Job $SLURM_JOB_ID | Part {idx}/{len(chunks)} | $(date) ==="
echo "Configs: {' '.join(map(str, chunk))}"

for CFG_ID in {' '.join(map(str, chunk))}; do
    OUTFILE="{corr_dir}/{ens}_Dpi_cfg${{CFG_ID}}_matrix_n64_ntsrc8_$(date +%Y-%m-%d).h5"
    if [[ -f "$OUTFILE" ]]; then
        echo "Skipping cfg $CFG_ID — output already exists"
        continue
    fi
    echo "Launching cfg $CFG_ID → $OUTFILE"
    srun --exact --cpus-per-task=$SLURM_CPUS_PER_TASK \\
         python3 -u $SRC_PATH --yaml_file "$YAML_FILE" --cfg_id $CFG_ID --outdir $CORR_DIR\\
         > {log_dir}/cfg_${{CFG_ID}}.log 2>&1 &
done

wait
echo "Part {idx} finished — correlators in {corr_dir}"
""")

        script_path.chmod(0o755)
        print(f"Generated {script_path.name} → {len(chunk)} configs")

    print(f"\nDone. Launch with:")
    print(f"   sbatch {run_dir}/contract_{ens}_part*.sh")
    print(f"   (or: for f in {run_dir}/*.sh; do sbatch \"$f\"; done)")
    print(f"\nFinal correlators will be in:\n   {corr_dir}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ini', type=str, required=True)
    parser.add_argument('--chunk', type=int, default=12, help="Configs per job")
    args = parser.parse_args()
    generate_batch_scripts(args.ini, args.chunk)
