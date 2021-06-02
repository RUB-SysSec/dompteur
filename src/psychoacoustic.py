import argparse
import shutil
import time
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from datasets import Dataset
from kaldi import Kaldi


BASE_DIR = Path.home().joinpath('dompteur')

def main(data_in_dir, data_out_dir, phi):

    print(f'\n[+] Process {data_in_dir}')

    if data_out_dir.is_dir():
        print(f"\n[!] Output dir {data_out_dir.name} already exists")
        return

    with TemporaryDirectory() as kaldi_dir:
        # copy data
        tmp_data_dir = Path(kaldi_dir).joinpath('tmp_data')
        shutil.copytree(data_in_dir, tmp_data_dir)
        # process wavs in container
        kaldi_tmp = Kaldi(Path(kaldi_dir))
        running_time = kaldi_tmp.run_in_container(
            f'python3 psycho/pre-processing.py --encoding_dir /root/tmp_data',
            additional_cmds=[f'-v {tmp_data_dir}:/root/tmp_data', 
                             f'-e PHI={phi}',                            
                             f'-e NUMJOBS={cpu_count()}']
        )
        # copy results from container
        shutil.copytree(tmp_data_dir, data_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_in_dir', type=Path, required=True, 
                        help='Directory with wavs.')
    parser.add_argument('--data_out_dir', type=Path, required=True,
                        help='Output directory for processed wavs.')
    parser.add_argument('--phi', default="0", 
                        help='Scaling factor for the psychoacoustic filter.')

    Kaldi.build_container(BASE_DIR)
    main(**vars(parser.parse_args()))
