import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
from scipy.io import wavfile


class PersistentDefaultDict:

    """
    Nested defaultdict that gets synced transparently to disk.

    Init: 
        results = PersistentDefaultDict(<path_to_results_file>)

    Add result:
        results['key1', 'key2'] = <result>
    """    

    def __init__(self, path_to_dict):
        self.path = Path(path_to_dict)
        if self.path.is_file():
            stored_data = json.loads(self.path.read_text())
            self.data = PersistentDefaultDict.redefault_dict(stored_data)
        else:
            self.data = defaultdict(PersistentDefaultDict.rec_default_dict)

    def __str__(self):
        return str(json.dumps(self.data, indent=4))
            
    def __setitem__(self, keys, item):
        d = self.data
        if isinstance(keys, str):
            d[keys] = item
        elif isinstance(keys, tuple):
            for key in keys[:-1]:
                d = d[key]
            d[keys[-1]] = item
        else:
            raise NotImplementedError()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=4))

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return self.data.__iter__()

    @staticmethod
    def rec_default_dict():
        return defaultdict(PersistentDefaultDict.rec_default_dict)
        
    @staticmethod
    def redefault_dict(data):
        if isinstance(data, dict):
            return defaultdict(PersistentDefaultDict.rec_default_dict, {k: PersistentDefaultDict.redefault_dict(v) for k, v in data.items()})
        else:
            return data

def get_threshs(threshs_file):
    assert threshs_file.is_file()
    # read in hearing thresholds
    thresholds = Path(threshs_file).read_text()
    thresholds = [row.split(',') for row in thresholds.split('\n')]
    # remove padded frames (copies frames at end and beginning)
    thresholds = np.array(thresholds[4:-4], dtype=float)[:,:256]
    thresholds = torch.tensor(thresholds, dtype=torch.float32).to('cpu')
    thresholds = thresholds.permute((1,0))
    return thresholds

def plot_stats(phi, data_dir, filename):

    cmap = plt.get_cmap('viridis')

    def set_spectogram(axis, signal, sample_rate, colorbar=False):
        frequencies, times, spectrogram = scipy.signal.spectrogram(signal, sample_rate)
        im = axis.pcolormesh(times, frequencies, np.log(spectrogram), shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axis)
        # if colorbar:
        #     fig.colorbar(im, ax=axs.ravel().tolist())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data_dir = Path(data_dir).joinpath('data')
        sample_rate = 16000

        _, ae = wavfile.read(data_dir.joinpath(f'{filename}_ae.original.wav'))
        _, ae_filtered = wavfile.read(data_dir.joinpath(f'{filename}_ae.wav'))

        _, ref = wavfile.read(data_dir.joinpath(f'{filename}_ref.original.wav'))
        ref = ref[:ae.size]
        _, ref_filtered = wavfile.read(data_dir.joinpath(f'{filename}_ref.wav'))
        ref_filtered = ref_filtered[:ae.size]

        fig, axs = plt.subplots(2, 3, figsize=(12,8))

        _, _, spectrogram = scipy.signal.spectrogram(ref, sample_rate)
        vmin = np.log(spectrogram.min())
        vmax = np.log(spectrogram.max())

        axs[0, 0].set_title('Reference')
        axs[0, 1].set_title('+ Adv. Noise')
        axs[0, 2].set_title('= Adv. Example')
        set_spectogram(axs[0, 0], ref, sample_rate, colorbar=True)
        set_spectogram(axs[0, 1], ae-ref, sample_rate)
        set_spectogram(axs[0, 2], ae, sample_rate)

        if phi != "None":
            threshs = get_threshs(data_dir.joinpath(f'{filename}_ae.csv'))
            axs[1, 0].set_title('AE Hearing Thresh')
            axs[1, 1].set_title('AE Hearing Thresh Scaled')
            axs[1, 2].set_title('Masked Adv. Example')
            im = axs[1,0].pcolormesh(threshs, shading='auto', cmap=cmap, vmin=threshs.min(), vmax=threshs.max()+int(phi)); fig.colorbar(im, ax=axs[1,0])
            im = axs[1,1].pcolormesh(threshs+int(phi), shading='auto', cmap=cmap, vmin=threshs.min(), vmax=threshs.max()+int(phi)); fig.colorbar(im, ax=axs[1,1])
            set_spectogram(axs[1, 2], ae_filtered, sample_rate)

        # for ax in axs.flat:
        #     ax.label_outer()
        
        fig.savefig(data_dir.parent.joinpath(f'{filename}.png'))
