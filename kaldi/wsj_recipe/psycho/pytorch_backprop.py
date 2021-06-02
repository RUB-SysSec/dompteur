import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

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

def main(data_in_file, data_out_file, grad_file, threshs_file_id):
    print(f'[+] backpropagate')
    print(f"    -> threshs_file_id : {threshs_file_id}")
    print(f'    -> data_in_file    : {data_in_file}')
    print(f'    -> data_out_file   : {data_out_file}')
    print(f'    -> grad_file       : {grad_file}')
    print(f"    -> phi             : {PHI}")

    # get debug data
    # shutil.copyfile(grad_file,  Path(f'/root/experiment/{threshs_file_id}_grad_in.csv'))
    # shutil.copyfile(data_in_file,  Path(f'/root/experiment/{threshs_file_id}_data_in.csv'))
    # shutil.copyfile(data_out_file,  Path(f'/root/experiment/{threshs_file_id}_data_out.csv'))
    # shutil.copyfile(THRESHS_TMP.joinpath(f'threshs_id.{threshs_file_id}.csv'),  
    #                 Path(f'/root/experiment/{threshs_file_id}_threshs.csv'))
    # time.sleep(1000)

    data_in = np.genfromtxt(data_in_file, delimiter=',')
    data_dim = data_in.shape  
    data_in = data_in.reshape(-1)
    data_in = torch.Tensor(data_in)

    gradient_in = np.genfromtxt(grad_file, delimiter=',')
    gradient_in_shape = gradient_in.shape
    gradient_in = gradient_in.reshape(-1)
    gradient_in = torch.Tensor(gradient_in)
    log_signal_data("GRADIENT IN", gradient_in)

    data_out = np.genfromtxt(data_out_file, delimiter=',')
    data_out = data_out.reshape(-1)
    data_out = torch.Tensor(data_out)
    log_signal_data("DATA OUT", data_out)

    if  PHI is None:
        gradient_out = gradient_in.reshape(gradient_in_shape)

    else:
        threshs_file = THRESHS_TMP.joinpath(f'threshs_id.{threshs_file_id}.csv')
        data_in.requires_grad = True
        signal_out = Psycho(PHI).forward(data_in, threshs_file)
        signal_out.backward(gradient_in)
        log_signal_data("SIGNAL OUT", signal_out)
        gradient_out = data_in.grad.reshape(gradient_in_shape)

    log_signal_data("GRADIENT OUT", gradient_out)
    np.savetxt(grad_file, gradient_out, delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_in_file', type=Path)
    parser.add_argument('--data_out_file', type=Path)
    parser.add_argument('--grad_file', type=Path)
    parser.add_argument('--threshs_file_id', type=int)
    main(**vars(parser.parse_args()))
