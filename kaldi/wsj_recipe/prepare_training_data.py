from pathlib import Path
from subprocess import run
import os

from tqdm import tqdm
import shutil
from psycho.psycho import Psycho
from multiprocessing import Pool, cpu_count

PHI = int(os.environ["PHI"]) if os.environ["PHI"] != "None" else None
NUMJOBS = int(os.environ["NUMJOBS"])

def process_entry(entry):
    # assert that wav is send to stdout
    assert entry.endswith('|')
    # parse entry
    utterance = entry.split(' ')[0]    # extract utterance 
    wav_cmd = entry[len(utterance)+1:] # extract path to wav
    # convert wav
    wav_path = dataset_data_dir.joinpath(utterance).with_suffix(".wav")
    run(f'{wav_cmd[:-1]} > {wav_path}', shell=True)
    # convert
    if PHI is not None:
        threshs_file = Path(f'/root/WSJ_threshs/{utterance}.csv')
        out_file = Path(wav_path)
        in_file = wav_path.with_suffix('.original.wav')
        wav_path.rename(in_file)
        Psycho(PHI).convert_wav(in_file, threshs_file, out_file)

    # return updated entry
    return f"{utterance} {wav_path} \n"

if __name__ == "__main__":
    print(f'PREPARE TRAINING DATA')
    print(f"[+] parsed arguments")
    print(f"    -> phi     : {PHI}")
    print(f"    -> numjobs : {NUMJOBS}")
    
    # first, get paths of the datasets wav lists
    # -> for each dataset (e.g., test_dev93, train_si284, ...), 
    #    speech files are accessed via path stored in 'wav.scp'
    # -> skip 'local/*' as these are not further used
    data_dir = Path('data')
    datasets = sorted([ path for path in data_dir.glob('**/*.scp')
                            if not path.match('local/data/*.scp') ])
    for dataset in datasets:
        print(f"[+] {dataset}", end=" ")
        dataset_data_dir = dataset.parent.joinpath('data')
        if dataset_data_dir.is_dir(): shutil.rmtree(dataset_data_dir)
        dataset_data_dir.mkdir()
        entries = [ entry.strip() for entry in dataset.read_text().splitlines() if entry.strip() ]
        print(f'({len(entries)} wavs) ')
        with Pool(NUMJOBS) as p:
            updated_entries = [ e for e in tqdm(p.imap(process_entry, entries)) ]
        # update wav.scp
        dataset.write_text("".join(updated_entries))

