

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from psycho import Psycho
from subprocess import run, DEVNULL
from multiprocessing import Pool
from tqdm import tqdm 

import shutil
import os

PHI = int(os.environ["PHI"]) if os.environ["PHI"] != "None" else None
NUMJOBS = int(os.environ["NUMJOBS"])

def preprocess_wav(in_file):
    if PHI is not None:
        Psycho.calc_thresholds(in_file)
    threshs_file = in_file.with_suffix(".csv")
    out_file = Path(in_file)
    in_file.rename(in_file.with_suffix('.original.wav'))
    in_file = in_file.with_suffix('.original.wav')
    print(f"    convert {in_file.name} into {out_file.name}")
    Psycho(PHI).convert_wav(in_file, threshs_file, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding_dir', required=True)
    params = parser.parse_args()
    data_dir = Path(params.encoding_dir)
    print(f"[+] parsed arguments")
    print(f"    -> data   : {data_dir}")
    print(f"    -> phi    : {PHI}")
    print(f"    -> numjobs: {NUMJOBS}")

    # get threshs
    with Pool(NUMJOBS // 3) as p:
        wav_files = list(data_dir.glob("*.wav"))
        list(p.imap_unordered(preprocess_wav, wav_files))
