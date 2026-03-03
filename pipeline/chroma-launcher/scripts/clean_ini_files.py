import argparse
import os
import glob

def clean_files(file_type, base_dir):
    if file_type not in ['eigs', 'peram','peramclov','meson']:
        print("Invalid file type. Please specify 'eigs' 'perams' or 'meson'.")
        return

    cfg_dirs = glob.glob(os.path.join(base_dir, 'cnfg*'))
    for cfg_dir in cfg_dirs:
        nvec_dirs = glob.glob(os.path.join(cfg_dir,'numvec*'))

        for nvec_dir in nvec_dirs:
            xml_pattern = os.path.join(nvec_dir, f'{file_type}_*.ini.xml')
            sh_pattern = os.path.join(nvec_dir, f'{file_type}_*.sh')

            xml_files = glob.glob(xml_pattern)
            for xml_file in xml_files:
                try:
                    os.remove(xml_file)
                    print(f'Removed {xml_file}')
                except OSError as e:
                    print(f'Error removing {xml_file}: {e}')

            sh_files = glob.glob(sh_pattern)
            for sh_file in sh_files:
                try:
                    os.remove(sh_file)
                    print(f'Removed {sh_file}')
                except OSError as e:
                    print(f'Error removing {sh_file}: {e}')

def main():
    parser = argparse.ArgumentParser(description='Clean XML and shell script files from configuration directories.')
    parser.add_argument('file_type', type=str, help="Type of files to remove ('eigs' 'perams' 'meson')")
    parser.add_argument('--base_dir', type=str, default='.', help='Base directory containing the configuration directories (default: current directory)')
    args = parser.parse_args()

    clean_files(args.file_type, args.base_dir)

if __name__ == '__main__':
    main()

