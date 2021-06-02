import argparse
import csv
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool, TimeoutError
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('destdir', type=str, help='audiofile destination dir')
    parser.add_argument('adversarialdir', type=str, help='adversarial .csv dir')
    parser.add_argument('datadir', type=str, help='data/folder to read wav.scp')
    parser.add_argument('fs', type=int, help='sampling frequency')
    parser.add_argument('winlen', type=int, help='sampling frequency')
    args = parser.parse_args()

    destdir = args.destdir
    adversarialdir = args.adversarialdir
    datadir = args.datadir
    fs = args.fs
    winlen = args.winlen
    audiofile = []
    utt_id = []
    with open(datadir, 'r') as f:
        for line in f:
            line.strip()
            # get wav dir
            line = line.split()
            utt_id.append(line[0])
            # remove dir and extension, only keep file name
            line = os.path.basename(line[1])
            audiofile.append(line.split('.')[0])

    # read in .csv of each utternce, reshape and save as audio file
    for i, id in enumerate(utt_id):
        with open(os.path.join(adversarialdir, id, 'adversarial.csv'), 'r') as csvfile:
            utterance = np.asarray(list(csv.reader(csvfile)))[:,:-1]
            utterance = np.round(np.float32(utterance))
            if utterance.shape[1] != winlen:
                raise Exception('Wrong Shape for utterance: ' + os.path.join(adversarialdir, id, 'adversarial.csv'))

            audio_flat = np.reshape(utterance, -1)
            out_file = os.path.join(destdir, audiofile[i] + '.wav')
            print('[+] Synthesize ' + out_file)
            wavfile.write(out_file, fs, np.int16(audio_flat))

if __name__ == "__main__":
    main()