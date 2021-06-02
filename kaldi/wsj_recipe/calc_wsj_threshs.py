import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from psycho.psycho import Psycho
from subprocess import run, DEVNULL
from multiprocessing import Pool, cpu_count
from tqdm import tqdm 

import shutil

OUTPUT_DIR = Path('data/thresholds')

def calc_thresholds(in_file):
    out_file = in_file.with_suffix(".csv")
    with TemporaryDirectory() as tmp_dir:
        # copy wav in tmp dir
        tmp_wav_file = Path(tmp_dir).joinpath(in_file.name)
        shutil.copyfile(in_file, tmp_wav_file)
        # creat wav.scp
        tmp_wav_scp = Path(tmp_dir).joinpath('wav.scp')
        tmp_wav_scp.write_text(f'data {tmp_wav_file}\n')
        # get hearing threshs
        run(f"/root/hearing_thresholds/run_calc_threshold.sh /usr/local/MATLAB/MATLAB_Runtime/v96 {tmp_wav_scp} 512 256 {tmp_dir}/", 
             stdout=DEVNULL, stderr=DEVNULL, shell=True)
        shutil.copyfile(Path(tmp_dir).joinpath('data_dB.csv'), out_file)

def process_entry(entry):
    # assert that wav is send to stdout
    assert entry.endswith('|')

    # parse entry
    utterance = entry.split(' ')[0]    # extract utterance 
    wav_cmd = entry[len(utterance)+1:] # extract path to wav
    if OUTPUT_DIR.joinpath(f'{utterance}.wav').is_file():
        # print(utterance)
        # some wavs are included twice (cf. dev_dt_05 / dev_dt_20 )
        return

    # convert wav
    wav_path = OUTPUT_DIR.joinpath(f'{utterance}.wav')
    run(f'{wav_cmd[:-1]} > {wav_path}', shell=True)

    # calc threshs
    calc_thresholds(wav_path)

if __name__ == "__main__":

    run("./import_wsj.sh")

    # make output dir
    if OUTPUT_DIR.is_dir(): shutil.rmtree()
    OUTPUT_DIR.mkdir()

    # first, get paths of the datasets wav lists
    # -> for each dataset (e.g., test_dev93, train_si284, ...), 
    #    speech files are accessed via path stored in 'wav.scp'
    # -> skip 'local/*' as these are not further used
    data_dir = Path('data')
    datasets = sorted([ path for path in data_dir.glob('**/*.scp')
                             if not path.match('local/data/*.scp') ], reverse=True)
    for dataset in datasets:
        print(f"[+] {dataset} ", end='')
        dataset_data_dir = dataset.parent.joinpath('data')
        entries = [ entry.strip() for entry in dataset.read_text().splitlines() if entry.strip() ]
        print(f'({len(entries)} wavs)')
        with Pool(cpu_count() // 3) as p:
            list(tqdm(p.imap_unordered(process_entry, entries)))