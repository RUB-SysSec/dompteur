import argparse
import time
from pathlib import Path

import numpy as np
import pydng

from datasets import Dataset
from kaldi import Kaldi
from select_models import select_model

BASE_DIR = Path.home().joinpath('dompteur')

def main(models, experiments, dataset_dir, phi, low, high):
    # create kaldi instance
    model_dir = select_model(models, phi, low, high)
    kaldi = Kaldi.from_trained_model(model_dir=model_dir,
                                     base_dir=experiments.joinpath(f'{time.strftime("%Y-%m-%d")}_{pydng.generate_name()}'))

    # prepare dataset
    dataset = Dataset(dataset_dir)
    kaldi_dataset_dir = kaldi.base_dir.joinpath("data", dataset.name)
    dataset.dump_as_kaldi_dataset(kaldi_dataset_dir, wavs_prefix=f'data/{dataset.name}')

    # decode
    wer, meta = kaldi.decode_wavs(data_dir=dataset.data_dir, text=dataset.text)
    print(f'\n[+] WER {dataset}: {wer:03.2f}%')
    for utt in meta:
        print(f"\n[+] {utt['wav_name']}")
        print(f"    REF: {utt['ref']}")
        print(f"    HYP: {utt['hyp']}")
        print(f"    WER: {utt['wer']*100:5.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=BASE_DIR.joinpath('models'),
                        help='Directory with trained models.')
    parser.add_argument('--experiments', type=Path, default=BASE_DIR.joinpath('experiments'),
                        help='Output directory for experiments.')
    parser.add_argument('--dataset_dir', type=Path, default=BASE_DIR.joinpath('datasets', 'speech_10'),
                        help='Path to dataset.')
    parser.add_argument('--phi', default="None",
                        help='Scaling factor for the psychoacoustic filter.')
    parser.add_argument('--low', default="None",
                        help='Lower cut-off frequency of band-pass filter.')
    parser.add_argument('--high', default="None",
                        help='Higher cut-off frequency of band-pass filter.')

    Kaldi.build_container(BASE_DIR)
    main(**vars(parser.parse_args()))
