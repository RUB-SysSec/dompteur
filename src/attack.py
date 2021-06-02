import argparse
import json
import shutil
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pydng

from datasets import Dataset
from kaldi import Kaldi
from kaldi_utils import KaldiLogger, parse_results_file
from select_models import select_model
from snr import snrseg
from utils import plot_stats

BASE_DIR = Path.home().joinpath('dompteur')

def main(models, experiments, dataset_dir, inner_itr, max_itr, learning_rate, psycho_hiding_thresh, phi, low, high, attacker):
    # create kaldi instance
    model_dir = select_model(models, phi, low, high)
    kaldi = Kaldi.from_trained_model(model_dir=model_dir,
                                     base_dir=experiments.joinpath(f'{time.strftime("%Y-%m-%d")}_{pydng.generate_name()}'))

    # prepare dataset
    dataset = Dataset(dataset_dir)
    kaldi_dataset_dir = kaldi.base_dir.joinpath("data", dataset.name)
    dataset.dump_as_kaldi_dataset(kaldi_dataset_dir, wavs_prefix=f'data/{dataset.name}')

    # optimization steps
    max_outer_itr = max_itr // inner_itr

    # dump config
    print(f'\n[+] Compute adversarial examples')
    print(f'    -> attacker "{attacker}"')
    print(f'    -> dataset "{dataset}"')
    print(f'    -> psycho_hiding_thresh "{psycho_hiding_thresh}dB"')
    print(f'    -> {max_outer_itr} * {inner_itr} = {inner_itr*max_outer_itr} itr')
    print(f'    -> phi={phi} bandpass={low}-{high}')
    print(f'    -> learning_rate={learning_rate}')
    kaldi.results['ae_config'] = {
        'phi' : phi,
        'low' : low,
        'high': high,
        'learning_rate' : learning_rate,
        'attacker' : attacker,
        'dataset' : dataset.name,
        'psycho_hiding_thresh' : psycho_hiding_thresh,
        'inner_itr' : inner_itr,
        'max_outer_itr' : max_outer_itr
    }

    # decode
    wer, _ = kaldi.decode_wavs(data_dir=dataset.data_dir, text=dataset.target)
    print(f'    -> inital WER: {wer:03.2f}%')
    kaldi.results['inital_wer'] = f'{wer:03.2f}%'

    # psychoacoustic filter causes an unstable gradient for backpropping to raw audio 
    # => caused by small input values to the log component
    # => thus, for backprop we clamp input before the log component (@ nnet-component.cc)
    log_clamp = 1 if phi != "None" else 0

    # invoke adversarial examples script
    logger = KaldiLogger(kaldi.base_dir)
    logger.log_ae(inner_itr, max_outer_itr)
    try:
        running_time = kaldi.run_in_container(
            f'./compute_adversarial_examples.sh {dataset} {psycho_hiding_thresh} {inner_itr} '
            f'{max_outer_itr} {len(dataset)} {len(dataset)} {attacker}',
            additional_cmds=[f'-v {kaldi_dataset_dir.joinpath("target_utterances")}:/root/kaldi/wsj_recipe/targets',
                             f'-e NUMJOBS={len(dataset)}',  
                             f'-e PHI={phi} -e LOG_CLAMP={log_clamp} -e LEARNING_RATE={learning_rate}']
        )
        logger.stop()
        kaldi.results['running_time'] = f'{running_time // 3600}h {(running_time % 3600) // 60}m {(running_time % 60)}s'
        print(f"    completed in {kaldi.results['running_time']}")
    except KeyboardInterrupt:
        logger.stop()
        print("    terminated prematurely")

    # score AEs
    kaldi.results['history'] = parse_results_file(kaldi.base_dir)
    score_AEs(kaldi=kaldi, 
              ae_dir=kaldi.base_dir.joinpath(f"adversarial_examples/wavs"), 
              ref_dir=dataset.data_dir,
              stats_dir=kaldi.base_dir.joinpath(f"adversarial_examples/stats"),
              original_text=dataset.text,
              target_text=dataset.target,
              phi=phi)


def score_AEs(kaldi, ae_dir, ref_dir, stats_dir, original_text, target_text, phi):
    print(f'\n[+] Score adversarial examples')

    # decode and score adversarial examples
    if stats_dir.is_dir():
        print(f'    stats dir already exists')
        return

    kaldi.fix_permissions()

    # -> preprocessing + decoding with target text
    wer, meta = kaldi.decode_wavs(data_dir=ae_dir, text=target_text, phi=phi)
    print(f'    -> WER             : {wer:03.2f}%')
    kaldi.results['wer'] = wer

    # -> preprocessing + decoding with original text
    wer, _ = kaldi.decode_wavs(data_dir=ae_dir, text=original_text, phi=phi)
    print(f'    -> WER Recovered   : {wer:03.2f}%')
    kaldi.results['wer_recovered'] = wer

    # copy successful AEs
    successful_AEs = [ ae['wav_name'] for ae in meta if ae['wer'] == 0 ]
    successful_AEs_dir = stats_dir.joinpath("successful_AEs")
    successful_AEs_dir.mkdir(parents=True)
    for successful_AE in successful_AEs:
        shutil.copy(ae_dir.joinpath(successful_AE).with_suffix('.wav'),
                    successful_AEs_dir.joinpath(successful_AE).with_suffix('.wav'))
    print(f'    -> Successful AEs  : {len(successful_AEs)} / {len(meta)}')
    kaldi.results['successful_AEs', 'count'] = len(successful_AEs)

    if len(successful_AEs) > 0: 
        # SNRseg
        snrs = snrseg(successful_AEs_dir, ref_dir)
        print(f'       SNRseg          : {np.mean(snrs):02.2f} (+-{np.std(snrs):02.2f})')
        kaldi.results['successful_AEs', 'snrseg'] = f'{np.mean(snrs):02.2f} (+-{np.std(snrs):02.2f})'

        # spectograms
        plots_dir = successful_AEs_dir.joinpath("plots")
        plots_dir.mkdir()
        with TemporaryDirectory() as kaldi_dir:
            tmp_data_dir = Path(kaldi_dir).joinpath('tmp_data')
            tmp_data_dir.mkdir()
            for successful_AE in successful_AEs:
                shutil.copyfile(
                    src=ae_dir.joinpath(f'{successful_AE}.wav'),
                    dst=tmp_data_dir.joinpath(f'{successful_AE}_ae.wav')
                )
                shutil.copyfile(
                    src=ref_dir.joinpath(f'{successful_AE}.wav'),
                    dst=tmp_data_dir.joinpath(f'{successful_AE}_ref.wav')
                )
            kaldi_tmp = Kaldi(Path(kaldi_dir))
            running_time = kaldi_tmp.run_in_container(
                f'python3 psycho/pre-processing.py --encoding_dir /root/tmp_data',
                additional_cmds=[f'-v {tmp_data_dir}:/root/tmp_data', 
                                f'-e PHI={phi}',                            
                                f'-e NUMJOBS={cpu_count()}']
            )
            shutil.copytree(tmp_data_dir, plots_dir.joinpath('data'))
        with Pool(len(successful_AEs)) as p:
            plot_func = partial(plot_stats, phi, plots_dir)
            list(p.imap(plot_func, successful_AEs))

    # copy unsuccessful AEs
    unsuccessful_AEs = [ ae['wav_name'] for ae in meta if ae['wer'] != 0 ]
    unsuccessful_AEs_dir = stats_dir.joinpath("unsuccessful_AEs")
    unsuccessful_AEs_dir.mkdir(parents=True)
    for unsuccessful_AE in unsuccessful_AEs:
        shutil.copy(ae_dir.joinpath(unsuccessful_AE).with_suffix('.wav'),
                    unsuccessful_AEs_dir.joinpath(unsuccessful_AE).with_suffix('.wav'))
    print(f'    -> Unsuccessful AEs: {len(unsuccessful_AEs)} / {len(meta)}')
    kaldi.results['unsuccessful_AEs', 'count'] = len(unsuccessful_AEs)

    if len(unsuccessful_AEs) > 0: 
        # SNRseg
        snrs = snrseg(unsuccessful_AEs_dir, ref_dir)
        print(f'       SNRseg          : {np.mean(snrs):02.2f} (+-{np.std(snrs):02.2f})')
        kaldi.results['unsuccessful_AEs', 'snrseg'] = f'{np.mean(snrs):02.2f} (+-{np.std(snrs):02.2f})'

        # spectograms
        plots_dir = unsuccessful_AEs_dir.joinpath("plots")
        plots_dir.mkdir()
        with TemporaryDirectory() as kaldi_dir:
            tmp_data_dir = Path(kaldi_dir).joinpath('tmp_data')
            tmp_data_dir.mkdir()
            for unsuccessful_AE in unsuccessful_AEs:
                shutil.copyfile(
                    src=ae_dir.joinpath(f'{unsuccessful_AE}.wav'),
                    dst=tmp_data_dir.joinpath(f'{unsuccessful_AE}_ae.wav')
                )
                shutil.copyfile(
                    src=ref_dir.joinpath(f'{unsuccessful_AE}.wav'),
                    dst=tmp_data_dir.joinpath(f'{unsuccessful_AE}_ref.wav')
                )
            kaldi_tmp = Kaldi(Path(kaldi_dir))
            running_time = kaldi_tmp.run_in_container(
                f'python3 psycho/pre-processing.py --encoding_dir /root/tmp_data',
                additional_cmds=[f'-v {tmp_data_dir}:/root/tmp_data', 
                                f'-e PHI={phi}',                            
                                f'-e NUMJOBS={cpu_count()}']
            )
            shutil.copytree(tmp_data_dir, plots_dir.joinpath('data'))
        with Pool(len(unsuccessful_AEs)) as p:
            plot_func = partial(plot_stats, phi, plots_dir)
            list(p.imap(plot_func, unsuccessful_AEs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=BASE_DIR.joinpath('models'),
                        help='Directory with trained models.')
    parser.add_argument('--experiments', type=Path, default=BASE_DIR.joinpath('experiments'),
                        help='Output directory for experiments.')
    parser.add_argument('--dataset_dir', type=Path, default=BASE_DIR.joinpath('datasets', 'speech_10'),
                        help='Path to target dataset.')
    parser.add_argument('--inner_itr', default=50, type=int,
                        help='Number of optimization steps in inner loop. After <inner_itr> steps, '
                             'AEs are deocded and hearing thresholds refreshed.')
    parser.add_argument('--max_itr', default=2000, type=int, 
                        help='Maximum number of optimization steps.')
    parser.add_argument('--learning_rate', default='0.05', 
                        help='Learning rate for the attack.')
    parser.add_argument('--phi', default="None",
                        help='Scaling factor for the psychoacoustic filter.')
    parser.add_argument('--low', default="None",
                        help='Lower cut-off frequency of band-pass filter.')
    parser.add_argument('--high', default="None",
                        help='Higher cut-off frequency of band-pass filter.')
    parser.add_argument('--attacker', default='adaptive', choices=['baseline', 'adaptive'],
                        help='Type of attacker.')
    parser.add_argument('--psycho_hiding_thresh', default="-1",
                        help='Margin "lambda" in dB for psychoacoustic hiding. Disabled for -1.')

    try:
        Kaldi.build_container(BASE_DIR)
        main(**vars(parser.parse_args()))
    finally:
        KaldiLogger.stop_all()
