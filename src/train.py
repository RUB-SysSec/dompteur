import argparse
import json
import secrets
import shutil
import time
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from kaldi import Kaldi
from kaldi_utils import KaldiLogger

BASE_DIR = Path.home().joinpath('dompteur')


def train_model(base_dir, wsj_dir, gpus='all', phi="None", low="None", high="None"):
    print(f'\n[+] train model phi={phi} bandpass={low}-{high} @ gpu {gpus}')
    print(f'    log @ {base_dir.joinpath("kaldi_log.txt")}')

    # sanity checks
    assert wsj_dir.joinpath('WSJ0').is_dir() and wsj_dir.joinpath('WSJ1').is_dir()
    if base_dir.is_dir(): 
        print(f'[!] {base_dir} already exists')
        return
    
    # hearing thresholds for WSJ dataset
    wsj_threshs_dir = wsj_dir.parent.joinpath('WSJ_threshs')
    if phi and not wsj_threshs_dir.is_dir():
        # psychoacoustic filter is enabled (i.e. phi != None) and
        # hearing thresholds do not yet exists
        # => pre-compute thresholds
        print(f"\n[+] pre-compute hearing thresholds for WSJ")
        with TemporaryDirectory() as kaldi_dir:
            print(f"    log @ {kaldi_dir}/kaldi_log.txt")
            kaldi_tmp = Kaldi(Path(kaldi_dir))
            running_time = kaldi_tmp.run_in_container(
                f'python3 calc_wsj_threshs.py',
                additional_cmds=[f'-v {wsj_dir}:/root/WSJ',                             
                                    f'-e NUMJOBS={cpu_count()}']
            )
            print(f"    completed in {running_time // 3600}h {(running_time % 3600) // 60}m {(running_time % 60)}s")
            shutil.copytree(kaldi_tmp.base_dir.joinpath("data/thresholds"), wsj_threshs_dir)
            print(f"    threshs @ {wsj_threshs_dir}")

    # calculcate start/end index for filter
    low_frequency_idx = int(np.ceil(int(low)/31.25) - 1) if low != "None" else 0
    high_frequency_idx = int(np.ceil(int(high)/31.25) - 1)if high != "None" else 255

    # training
    kaldi = Kaldi(base_dir)
    try:
        logger = KaldiLogger(base_dir)
        logger.log_training()
        running_time = kaldi.run_in_container(
            "./train_network.sh",
            additional_cmds=[f'-v {wsj_dir}:/root/WSJ', 
                             f'-v {wsj_threshs_dir}:/root/WSJ_threshs',
                             f'--gpus device={gpus}',
                             f'-e NUMJOBS={cpu_count()}',
                             f'-e PHI={phi} -e LOG_CLAMP=0',
                             f'-e LOW_FREQUENCY_IDX={low_frequency_idx} -e HIGH_FREQUENCY_IDX={high_frequency_idx}']
        )
        logger.stop()
        print(f'    completed in {running_time // 3600}h {(running_time % 3600) // 60}m {(running_time % 60)}s\n')
    except RuntimeError:
        print("    failed")
        logger.stop()
        return

    # get benign accuracy
    wer = benign_accuracy(kaldi, wsj_dir, phi)

    # save training log
    kaldi.log_file.rename(kaldi.base_dir.joinpath('training.log.txt'))

    # finally, clean up model
    kaldi.cleanup()


def benign_accuracy(kaldi, wsj_dir, phi):
    # compute the benign accuracy for the WSJ test set `eval92`
    # => first, convert WSJ to raw wavs
    wsj_raw_data_dir = wsj_dir.parent.joinpath('WSJ_raw')
    if not wsj_raw_data_dir.is_dir():
        wsj_raw_data_dir.mkdir()
        print(f"[+] convert WSJ to wavs")
        with TemporaryDirectory() as kaldi_dir:
            print(f"    log @ {kaldi_dir}/kaldi_log.txt")
            kaldi_tmp = Kaldi(Path(kaldi_dir))
            running_time = kaldi_tmp.run_in_container(
                f'./import_wsj.sh && python3 prepare_training_data.py',
                additional_cmds=[f'-v {wsj_dir}:/root/WSJ',
                                 f'-e NUMJOBS={cpu_count()}',
                                 f'-e PHI=None']
            )
            print(f"    completed in {running_time // 3600}h {(running_time % 3600) // 60}m {(running_time % 60)}s")
            datasets = sorted([ path for path in kaldi_tmp.base_dir.joinpath('data').glob('**/*.scp')
                                        if not path.match('local/data/*.scp') ])
            for dataset in datasets:
                dataset_dir = wsj_raw_data_dir.joinpath(dataset.parent.name)
                shutil.copytree(src=dataset.parent.joinpath('data'), dst=dataset_dir)
                text = { line.strip().split(' ', 1)[0] : line.strip().split(' ', 1)[1]
                            for line in dataset.parent.joinpath("text").read_text().splitlines()
                            if line.strip() }
                dataset_dir.joinpath('text.json').write_text(json.dumps(text, indent=4))
    # decode
    print(f"[+] benign accuracy for phi={phi}")
    data = wsj_raw_data_dir.joinpath("test_eval92")
    text = json.loads(wsj_raw_data_dir.joinpath("test_eval92/text.json").read_text())
    wer, _ = kaldi.decode_wavs(data, text, phi=phi)
    print(f'    -> WER: {wer:03.2f}%')
    return wer

def main(models, wsj_dir, phi, low, high, gpus):

    try:
        base_dir = models.joinpath(f'dompteur_{secrets.token_hex(2)}_phi.{phi}_bandpass.{low}-{high}')
        train_model(base_dir=base_dir, wsj_dir=wsj_dir, 
                    gpus=gpus, phi=phi, low=low, high=high)
    except RuntimeError:
        print("\n    failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=BASE_DIR.joinpath('models'),
                        help='Directory for trained models.')
    parser.add_argument('--wsj_dir', type=Path, default=BASE_DIR.joinpath('WSJ'),
                        help='Path to Wall Street Journal (WSJ) specch corpus.')
    parser.add_argument('--phi', default="None",
                        help='Scaling factor for the psychoacoustic filter.')
    parser.add_argument('--low', default="None",
                        help='Lower cut-off frequency of band-pass filter.')
    parser.add_argument('--high', default="None",
                        help='Higher cut-off frequency of band-pass filter.')
    parser.add_argument('--gpus', default="all",
                        help='GPU devices used for training.')

    Kaldi.build_container(BASE_DIR)
    main(**vars(parser.parse_args()))
