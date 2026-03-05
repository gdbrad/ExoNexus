#!/usr/bin/env python3
import argparse
import subprocess
import sys
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run full analysis locally.")
    parser.add_argument("--run_dir", required=True, help="Path to the finished run directory (e.g., .../run-YYYYMMDD_X)")
    parser.add_argument("--yaml", required=True, help="Path to the ensemble YAML configuration file")
    parser.add_argument("--bad_idx", type=int, default=58, help="Index of bad config to drop (default 58 for cfg 3400)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    yaml_path = Path(args.yaml).resolve()
    
    # Extract ensemble name from yaml
    with open(yaml_path) as f:
        full_config = yaml.safe_load(f)
        ensemble_key = list(full_config.keys())[0]
        ens_name = full_config[ensemble_key]["ensemble"]["short"]
    
    # ---------------------------------------------------------
    # 1. Locate Scripts
    # ---------------------------------------------------------
    this_script_dir = Path(__file__).resolve().parent
    repo_root = this_script_dir.parent.parent
    
    analyze_script = repo_root / "analysis" / "analyze_spec.py"
    resample_script = repo_root / "contractions" / "postprocess" / "build_full_spec_resampled.py"
    
    if not resample_script.exists():
        print(f"Error: Could not find resampler at {resample_script}.")
        sys.exit(1)

    if not analyze_script.exists():
        print(f"Error: Could not find analysis script at {analyze_script}.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Run Stage-3 Resampling
    # ---------------------------------------------------------
    print("\n=============================================")
    print(f"Running Stage-3: {resample_script.name}")
    
    stage2_dir = run_dir / "stage2_cij"
    if not stage2_dir.exists():
        print(f"Error: {stage2_dir} does not exist. Ensure Stage 2 completed.")
        sys.exit(1)

    resample_cmd = [
        "python3", "-u", str(resample_script),
        "--data", str(stage2_dir),
        "--outdir", str(run_dir),
        "--ens", ens_name,
    ]
    
    if args.bad_idx is not None:
        resample_cmd.extend(["--bad_idx", str(args.bad_idx)])
    
    try:
        subprocess.run(resample_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nResampling failed with error: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. Locate Resampled Output & Run Stage-4 Analysis
    # ---------------------------------------------------------
    # Look for files matching 'spec_{ens}*.h5'
    spec_files = list(run_dir.glob(f"spec_{ens_name}_*.h5"))

    if not spec_files:
        print(f"\nError: Resampler finished, but no resampled spec .h5 files were found in {run_dir}")
        sys.exit(1)

    # Pick the most recently created one if multiple exist
    latest_spec = max(spec_files, key=lambda p: p.stat().st_mtime)

    print(f"\n=============================================")
    print(f"Running Stage-4 Analysis on: {latest_spec.name}")
    
    analyze_cmd = [
        "python3", "-u", str(analyze_script),
        "--specfile", str(latest_spec)
    ]

    try:
        subprocess.run(analyze_cmd, check=True)
        print(f"\nAnalysis successfully completed for {latest_spec.name}.")
        print(f"Plots and fits are saved in {latest_spec.parent}/plots and {latest_spec.parent}/fits.")
    except subprocess.CalledProcessError as e:
        print(f"\nAnalysis failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()