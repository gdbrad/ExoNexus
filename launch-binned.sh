#!/bin/bash

usage() {
    echo "Usage: $0 -e <ensemble> -n <numvec> -t <type> [-c <cfg_numbers>] [-s <skip_range>] [-r <tsrc>] [--convert] [--test]"
    echo "  -e <ensemble>   : Specify the ensemble (e.g., 'b3.6_s48t64',)"
    echo "  -n <numvec>     : Specify the numvec value (e.g., 32, 64, 128, etc.)"
    echo "  -t <type>       : Specify the type ('meson', 'peram', 'peram_strange', 'eigs')"
    echo "  -c <cfg_numbers>: Specify specific configuration numbers to process (comma-separated, e.g., '101,1031,1051')"
    echo "  -s <skip_range> : Specify the range of configurations to skip (e.g., '1961-1991')"
    echo "  -r <tsrc>       : Specify the number of tsrcs"
    echo "  --convert       : Run the HDF5 conversion instead of submitting jobs"
    echo "  --test          : Echo the commands that would be run, but don't execute them"
    exit 1
}

# Default values
convert=false
test=false
cfg_numbers=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e) ensemble="$2"
            shift 2
            ;;
        -n) numvec="$2"
            shift 2
            ;;
        -t) type="$2"
            shift 2
            ;;
        -c) IFS=',' read -r -a cfg_numbers <<< "$2"
            shift 2
            ;;
        -s) skip_range="$2"
            shift 2
            ;;
        -r) tsrc="$2"
            shift 2
            ;;
        --convert) convert=true
            shift
            ;;
        --test) test=true
            shift
            ;;
        *) usage
            ;;
    esac
done

# Check required arguments
if [ -z "$ensemble" ] || [ -z "$numvec" ] || [ -z "$type" ]; then
    usage
fi

# Validate type
if [ "$type" != "meson" ] && [ "$type" != "peram" ] && [ "$type" != "peram_charm" ] && [ "$type" != "eigs" ]; then
    echo "Invalid type specified. Please choose 'meson', 'peram', 'peram_strange', or 'eigs'."
    exit 1
fi

# Set BASE_DIR based on ensemble and type
BASE_DIR="/p/scratch/exotichadrons/su3-distillation/data/$ensemble"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Ensemble directory $BASE_DIR does not exist."
    exit 1
fi

# Chroma setup (unchanged)
CODE_DIR="/p/scratch/exotichadrons/chroma-distillation-superb"
source $CODE_DIR/install-scripts/machines/env-new-jureca-gpu.sh
CHROMA_BIN=$CODE_DIR/install/chroma/bin

export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1,2,3

check_file_size() {
    local file="$1"
    local min_size="$2"

    actual_size=$(stat -c%s "$file")
    if [ "$actual_size" -lt "$min_size" ]; then
        return 1
    else
        return 0
    fi
}

if [ "$convert" = true ]; then
    echo "Running HDF5 conversion for $type in ensemble $ensemble..."

    # Set SDB_DIR based on type and ensemble
    if [ "$type" = "peram" ]; then
        SDB_DIR="$BASE_DIR/perams_sdb/numvec$numvec"
    elif [ "$type" = "peram_charm" ]; then
         SDB_DIR="$BASE_DIR/perams_charm_sdb"

    elif [ "$type" = "eigs" ]; then
        SDB_DIR="$BASE_DIR/eigs_sdb"
    else
        SDB_DIR="$BASE_DIR/${type}_sdb"

    fi
    H5_DIR="/p/scratch/exotichadrons/su3-distillation/h5-out"
    # simple chroma input file for running conversions (just does a dummy plaquette measurement)
    in="$H5_DIR/simple.ini.xml"

    min_size=$((400 * 1024 * 1024))

    for sdb_file in "$SDB_DIR"/*.sdb; do
        if [ -f "$sdb_file" ]; then
            base_name=$(basename "$sdb_file" .sdb)
            if [ "$type" = "peram_strange" ]; then
                cfg_num=$(echo "$base_name" | awk -F'_' '{print $NF}')
            else
                cfg_num=$(echo "$base_name" | grep -oP '\d+$')
            fi

            if [[ "${cfg_numbers[*]}" =~ (^|[[:space:]])$cfg_num($|[[:space:]]) ]] || [ ${#cfg_numbers[@]} -eq 0 ]; then
                mkdir -p "$SDB_DIR/log-trash"
                mkdir -p "$SDB_DIR/xml-trash"
                mkdir -p "$SDB_DIR/stdout"
                
                log="$SDB_DIR/log-trash/${base_name}_log.xml"
                out="$SDB_DIR/xml-trash/${base_name}_out.xml"
                stdout="$SDB_DIR/stdout/${base_name}_convert.stdout"
                h5_file="$SDB_DIR/${base_name}.h5"

                if [ ! -f "$h5_file" ] || ([ "$type" = "meson" ] && [ "$numvec" -eq 32 ] && ! check_file_size "$h5_file" "$min_size"); then
                    if [ -f "$h5_file" ]; then
                        echo "Warning: $h5_file is smaller than expected size (400 MB)."
                        read -p "Do you want to remove and rerun the conversion for this file? (y/n): " choice
                        if [ "$choice" = "y" ]; then
                            rm "$h5_file"
                        else
                            continue
                        fi
                    fi

                    normalized_type="$type"
                    if [ "$type" = "peram_strange" ]; then
                        normalized_type="peram"
		    elif [ "$type" = "peram_charm" ]; then
                       normalized_type="peram"

                    fi

                    object="$CHROMA_BIN/convert_${normalized_type}$numvec $sdb_file"
                    export OPTS="-geom 1 1 1 1" # QIO does not support multi-node jobs
                    echo "running conversion for $sdb_file..."

                    if [ "$test" = true ]; then
                        echo "srun --account=exotichadrons --partition=dc-gpu -t 00:15:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --threads-per-core=1 -n 1 -c 1 $object $OPTS -i $in -o $out -l $log > $stdout 2>&1"
                    else
                        srun --account=exotichadrons --partition=dc-gpu-devel -t 00:10:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --threads-per-core=1 -n 1 -c 1 $object $OPTS -i $in -o $out -l $log > $stdout 2>&1
                    fi
                fi
            fi
        else
            echo "No .sdb files found in $SDB_DIR"
        fi
    done

    echo "FINISH JOB $(date "+%Y-%m-%dT%H:%M")"
    exit 0
fi

# Normal operation: Submit sdb generation jobs
echo "Submitting jobs for ensemble $ensemble..."

if [ ! -z "$skip_range" ]; then
    skip_start=$(echo $skip_range | cut -d'-' -f1)
    skip_end=$(echo $skip_range | cut -d'-' -f2)
fi

for cfg_dir in "$BASE_DIR"/cfgs_*; do
    if [ -d "$cfg_dir" ]; then
        dir_name=$(basename "$cfg_dir")
        dir_range=${dir_name#cfgs_}
        start_range=${dir_range%-*}
        end_range=${dir_range#*-}

        if [ ! -z "$skip_range" ]; then
            if [ "$start_range" -ge "$skip_start" ] && [ "$end_range" -le "$skip_end" ]; then
                echo "Skipping directory: $cfg_dir"
                continue
            fi
        fi

        echo "Processing directory: $cfg_dir"

        numvec_dir="$cfg_dir/numvec$numvec"

        if [ -d "$numvec_dir" ]; then
            echo "  Processing subdirectory: $numvec_dir"

            for script in "$numvec_dir"/${type}*; do
                if [ -f "$script" ]; then
                    cfg_num=$(echo $(basename "$script") | grep -oP '\d+$')

                    if [[ "${cfg_numbers[*]}" =~ (^|[[:space:]])$cfg_num($|[[:space:]]) ]] || [ ${#cfg_numbers[@]} -eq 0 ]; then
                        if [ "$test" = true ]; then
                            echo "sbatch $script"
                        else
                            echo "    Submitting script: $script"
                            sbatch "$script"
                        fi
                    fi
                fi
            done
        fi
    fi
done
