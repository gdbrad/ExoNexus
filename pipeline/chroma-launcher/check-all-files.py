import os
import argparse

# for beta in [3.4,3.6]:
#     dir_path = f"{beta}_s{s}t{}
    
DIRECTORIES =  { 
"meson":             "/p/scratch/exotichadrons/su3-distillation/meson_sdb/",
"peram":             "/p/scratch/exotichadrons/exolaunch/perams_sdb/numvec64/",
}

SDB_PATTERNS = {
    "meson": "meson-{}_cfg{}.sdb",
    "peram": "peram_{}_cfg{}.sdb",
}

H5_PATTERNS = {
    "meson": "meson-{}_cfg{}.h5",
    "peram": "peram_{}_cfg{}.h5",
}
nvecs = [96]

CFG_RANGE = range(11, 2000, 10)

def check_files(file_type):
    missing_sdb = []
    missing_h5 = []
    nvecs = [96]

    for nvec in nvecs:
        if file_type in DIRECTORIES:
            directory = DIRECTORIES[file_type].format(nvec)
            sdb_pattern = SDB_PATTERNS[file_type]
            h5_pattern = H5_PATTERNS[file_type]

            for cfg in CFG_RANGE:
                sdb_file = sdb_pattern.format(nvec, cfg)
                h5_file = h5_pattern.format(nvec, cfg)

                sdb_path = os.path.join(directory, sdb_file)
                h5_path = os.path.join(directory, h5_file)

                sdb_exists = os.path.isfile(sdb_path)
                h5_exists = os.path.isfile(h5_path)

                if not sdb_exists:
                    missing_sdb.append(sdb_path)
                if not h5_exists:
                    missing_h5.append(h5_path)
        else:
            print(f"Unknown file type: {file_type}")
            return

    return missing_sdb, missing_h5

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Check for missing SDB and HDF5 files.")
    parser.add_argument(
        "-t", "--type", 
        choices=["meson", "peram", "peram_strange", "all"], 
        required=True, 
        help="Specify the type to process: meson, peram, peram_strange, or all."
    )
    args = parser.parse_args()

    file_type = args.type

    if file_type == "all":
        types_to_check = ["meson", "peram", "peram_strange"]
    else:
        types_to_check = [file_type]

    for type_to_check in types_to_check:
        print(f"Processing type: {type_to_check}")
        missing_sdb, missing_h5 = check_files(type_to_check)

        if missing_sdb:
            print(f"\nMissing SDB files for {type_to_check}:")
            for file in missing_sdb:
                print(file)
        else:
            print(f"All SDB files are present for {type_to_check}.")

        if missing_h5:
            print(f"\nMissing HDF5 files for {type_to_check}:")
            for file in missing_h5:
                print(file)
        else:
            print(f"All HDF5 files are present for {type_to_check}.")

if __name__ == "__main__":
    main()
