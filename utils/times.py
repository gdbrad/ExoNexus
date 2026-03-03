
import os
import re
import shutil
from collections import defaultdict

def extract_time_from_file(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            match = re.search(r'CHROMA: total time = (\d+\.\d+) secs', line)
            if match:
                return float(match.group(1))
    return 0.0

def extract_cfg_id(filename,type):
    if type=='meson':
        match = re.search(r'cfg(\d+)|(\d+)_numvecs|meson(\d+)', filename)
    elif type=='perams':
        #perams_1561_numvecs-32.out
        match = re.search(r'cfg(\d+)|(\d+)_numvecs|perams(\d+)', filename)

    if match:
        return match.group(1) or match.group(2) or match.group(3)
    return None

def sum_times_and_count_files_in_directory(directory):
    total_time = 0.0
    file_count = 0
    cfg_files = defaultdict(list)
    duplicates = {}
    cfg_ids_present = set()

    for root, dirs, files in os.walk(directory):
        for file in files:
            # if '32' in file:  # Ensure the file has '32' in its name
            filepath = os.path.join(root, file)
            total_time += extract_time_from_file(filepath)
            file_count += 1

            cfg_id = extract_cfg_id(file)
            if cfg_id:
                cfg_files[cfg_id].append(filepath)
                cfg_ids_present.add(int(cfg_id))

    for cfg_id, files in cfg_files.items():
        if len(files) > 1:
            duplicates[cfg_id] = files

    return total_time, file_count, duplicates, cfg_ids_present

def convert_seconds_to_hours_minutes(seconds):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return int(hours), int(minutes)

def move_duplicate_files(duplicates,target_directory,nvecs,type):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    nvecs_str = str(nvecs)  # Convert nvecs to string for use in the regex

    for cfg_id, files in duplicates.items():
        for file in files:
            # Construct the regex pattern with the nvecs value
            if type  =='meson':
                pattern = rf'meson\d+_numvecs-{nvecs_str}\.out$'
            elif type=='perams':
            #perams_1561_numvecs-32.out

                pattern = rf'perams_\d+_numvecs-{nvecs_str}\.out$'

            if re.search(pattern, file):
                shutil.move(file, os.path.join(target_directory, os.path.basename(file)))
                print(f"Moved {file} to {target_directory}")


def check_missing_cfg_files(cfg_ids_present, start=11, end=2000, step=10):
    missing_cfg_ids = []
    for cfg_id in range(start, end, step):
        if cfg_id not in cfg_ids_present:
            missing_cfg_ids.append(cfg_id)
    return missing_cfg_ids


# Directory containing the files
# eigs_path = '/p/scratch/exotichadrons/exolaunch/chroma_out/eigs-stdout'
paths = [
'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec32',
'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec64',
'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec96',
'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec128',

'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec32',
'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec64',
'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec96',
'/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec128']




# Directory to move duplicate files to
# target_directory_path = '/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec64/duplicates'
# target_directory_path = '/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec96/duplicates'
target_directory_path = '/p/scratch/exotichadrons/exolaunch/chroma_out/meson-stdout/numvec128/duplicates'

# Summing the times, counting the files, finding duplicates, and checking cfg_ids present
# total_seconds, file_count, duplicates, cfg_ids_present = sum_times_and_count_files_in_directory(eigs_path)
# total_seconds, file_count, duplicates, cfg_ids_present = sum_times_and_count_files_in_directory(meson64_path)
# total_seconds, file_count, duplicates, cfg_ids_present = sum_times_and_count_files_in_directory(meson96_path)
for path in paths:
    total_seconds, file_count, duplicates, cfg_ids_present = sum_times_and_count_files_in_directory(path)



# total_seconds, file_count, duplicates, cfg_ids_present = sum_times_and_count_files_in_directory(directory_path)


    # Converting total time to hours and minutes
    hours, minutes = convert_seconds_to_hours_minutes(total_seconds)

    # Calculating the average time per file
    average_seconds = total_seconds / file_count if file_count > 0 else 0
    average_hours, average_minutes = convert_seconds_to_hours_minutes(average_seconds)

    print(f"Total time: {hours} hours and {minutes} minutes")
    print(f"Number of files: {file_count}")
    print(f"Average time per file: {average_hours} hours and {average_minutes} minutes")

    # Printing duplicates
    if duplicates:
        print("Duplicate files found:")
        for cfg_id, files in duplicates.items():
            print(f"Config ID {cfg_id} has the following files:")
            for file in files:
                print(f"  - {file}")
    else:
        print("No duplicate files found.")

# Moving duplicate files
#move_duplicate_files(duplicates, target_directory_path,nvecs=128)

    # Checking for missing configuration files
    missing_cfg_ids = check_missing_cfg_files(cfg_ids_present)

    if missing_cfg_ids:
        print("Missing configuration files for the following cfg IDs:")
        for cfg_id in missing_cfg_ids:
            print(f"  - cfg{cfg_id}")
    else:
        print("No missing configuration files found.")
