
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
from pathlib import Path
from tempfile import mkstemp

import numpy as np
import torch
from scipy.io import wavfile

from psycho import Psycho

KALDI_BASE = Path('/root/kaldi/wsj_recipe')
THRESHS_TMP = KALDI_BASE.joinpath('exp/threshs_tmp')
THRESHS_TMP.mkdir(exist_ok=True)
PHI = int(os.environ["PHI"]) if os.environ["PHI"] != "None" else None

def log_signal_data(name, signal):
    try:
        print(f'\n    {name.upper()}')
        print(f'    -> MIN  {signal.min()}')
        print(f'    -> MAX  {signal.max()}')
        print(f'    -> MEAN {signal.abs().mean()}')
        print(f'    -> VALS {signal[1000]} {signal[10000]} {signal[20000]}')
    except:
        pass

def main(data_file, threshs_file_id):
    print(f'[+] propagate')
    print(f"    -> threshs_file_id: {threshs_file_id}")
    print(f"    -> phi            : {PHI}")
    print(f'    -> data file      : {data_file}')
    
    # load data 
    data = np.genfromtxt(data_file, delimiter=',')
    data_dim = data.shape
    data = data.reshape(-1)
    data = torch.Tensor(data)
    log_signal_data('DATA IN', data)
    
    # pre-process
    threshs_file = THRESHS_TMP.joinpath(f'threshs_id.{threshs_file_id}.csv')
    if PHI is not None and not threshs_file.is_file():
        fd_tmp_in, tmp_in = mkstemp()
        input_signal_int16 = np.int16(np.round(data))
        wavfile.write(tmp_in, 16000, input_signal_int16)
        Psycho.calc_thresholds(tmp_in, out_file=threshs_file)
        os.close(fd_tmp_in)
        os.remove(tmp_in)

    # apply psychoacoustic thresholds
    signal_out = Psycho(PHI).forward(data, threshs_file)
    log_signal_data('DATA OUT', signal_out)

    # dump back to data file
    signal_out = signal_out.detach().numpy()
    signal_out = signal_out.reshape(data_dim)
    np.savetxt(data_file, signal_out, delimiter=',')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=Path)
    parser.add_argument('--threshs_file_id', type=int)
    main(**vars(parser.parse_args()))
