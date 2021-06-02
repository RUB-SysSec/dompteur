from pathlib import Path
import numpy as np
from scipy.io import wavfile
from tempfile import TemporaryDirectory
from subprocess import run, DEVNULL
import shutil
import torch
import torchaudio
import torch.nn.functional as F
torch.set_num_threads(1)

class Psycho:

    def __init__(self, phi):
        self.phi = phi
        self.sampling_rate = 16000
        self.win_length = 512 
        self.hop_length = 256

    @staticmethod
    def calc_thresholds(in_file, out_file=None):
        in_file = Path(in_file)
        if not out_file: out_file = in_file.with_suffix(".csv")
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

    def get_psycho_mask(self, complex_spectrum, threshs_file):
        tmp_complex_spectrum  = complex_spectrum.detach().clone()
        # Step 1: remove offset
        offset = tmp_complex_spectrum[0,:,:]
        features = tmp_complex_spectrum[1:,:,:]
        # Step 2: represent as phase and magnitude
        a_re = features[:,:,0]; a_re[torch.where(a_re == 0)] = 1e-20
        b_im = features[:,:,1]
        # phase
        phase = torch.atan( b_im / a_re )
        phase[torch.where(a_re < 0)] += np.pi
        # magnitude
        magnitude = torch.sqrt( torch.square(a_re) + torch.square(b_im) )
        # Step 3: get thresholds
        assert self.phi is not None
        # import thresholds
        assert threshs_file.is_file()
        # read in hearing thresholds
        thresholds = Path(threshs_file).read_text()
        thresholds = [row.split(',') for row in thresholds.split('\n')]
        # remove padded frames (copies frames at end and beginning)
        thresholds = np.array(thresholds[4:-4], dtype=float)[:,:256]
        thresholds = torch.tensor(thresholds, dtype=torch.float32)
        thresholds = thresholds.permute((1,0))
        # Step 4: calc mask
        m_max = magnitude.max()
        S = 20*torch.log10(magnitude / m_max) # magnitude in dB
        H = thresholds - 95
        # scale with phi
        H_scaled  = H + self.phi
        # mask 
        mask = torch.ones(S.shape)
        mask[torch.where(S <= H_scaled)] = 0
        mask_offset = torch.ones((1, mask.shape[1]))
        mask = torch.cat((mask_offset, mask), dim=0)
        mask = torch.stack((mask, mask), dim=2)
        return mask

    def forward(self, signal, threshs_file):

        if self.phi is None:
            return signal

        # fft
        complex_spectrum = torch.stft(signal, 
                            n_fft=self.win_length, 
                            hop_length=self.hop_length, 
                            win_length=self.win_length,
                            window=torch.hamming_window(self.win_length), 
                            pad_mode='constant', 
                            onesided=True)

        # mask signal with psychoacoustic thresholds 
        mask = self.get_psycho_mask(complex_spectrum, threshs_file)
        complex_spectrum_masked = complex_spectrum * mask
        
        # ifft
        signal_out = torch.istft(complex_spectrum_masked,
                    n_fft=self.win_length, 
                    hop_length=self.hop_length, 
                    win_length=self.win_length,
                    window=torch.hamming_window(self.win_length), 
                    onesided=True)

        return signal_out

    def convert_wav(self, in_file, threshs_file, out_file, device='cpu'):
        torch_signal, torch_sampling_rate = torchaudio.load(in_file)
        torch_signal = (torch.round(torch_signal*32767)).squeeze().to(device)
        signal_out = self.forward(torch_signal, threshs_file)
        signal_out = torch.round(signal_out).cpu().detach().numpy().astype('int16')
        wavfile.write(out_file, self.sampling_rate, signal_out)
