from pathlib import Path

import numpy as np
from scipy.io import wavfile


def numel(array):
    s = array.shape
    n = 1
    for i in range(len(s)):
        n *= s[i]
    return n

def _snrseg(noisy_file, clean_file, tf=0.05):
    # read wav
    fs, clean_signal = wavfile.read(clean_file)
    _, noisy_signal = wavfile.read(noisy_file)
    r = clean_signal.astype(np.float32, order='C') / 32768.0
    s = noisy_signal.astype(np.float32, order='C') / 32768.0
    # snr
    snmax = 100
    nr = min(r.shape[0], s.shape[0])
    kf = round(tf * fs)
    ifr = np.arange(kf, nr, kf)
    ifl = int(ifr[len(ifr)-1])
    nf = numel(ifr)
    ef = np.sum(np.reshape(np.square((s[0:ifl] - r[0:ifl])), (kf, nf), order='F'), 0)
    rf = np.sum(np.reshape(np.square(r[0:ifl]), (kf, nf), order='F'), 0)
    em = ef == 0
    rm = rf == 0
    snf = 10 * np.log10((rf + rm) / (ef + em))
    snf[rm] = -snmax
    snf[em] = snmax
    temp = np.ones(nf)
    vf = temp == 1
    seg = np.mean(snf[vf])
    return seg

def snrseg(noisy_dir, clean_dir):
    snrs = []
    for noisy_file in Path(noisy_dir).glob('*.wav'):
        clean_file = Path(clean_dir).joinpath(noisy_file.name)
        snrs.append(_snrseg(noisy_file, clean_file))
    return snrs
