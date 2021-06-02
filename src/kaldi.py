import json
import os
import random
import re
import secrets
import shutil
import time
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from datasets import Dataset
from kaldi_utils import *
from snr import *
from utils import *


class Kaldi:

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        base_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.base_dir.joinpath('kaldi_log.txt')
        self.results = PersistentDefaultDict(self.base_dir.joinpath(f'results.json'))
        self.fix_permissions()

    @staticmethod
    def build_container(base_dir):
        docker_log = base_dir.joinpath('docker.log.txt')
        while docker_log.is_file():
            # we use the log as a lock for building
            # => wait for a few seconds and try again
            r = random.randint(5, 30)
            print(f"[!] docker.log.txt exists.\n    -> wait for {r}s")
            time.sleep(r)
        print('[+] build container')
        print(f'    log @ {docker_log}')
        with open(docker_log, 'w') as log_file:
            if run(f"docker build -t dompteur {base_dir}", 
                   stdout=log_file, stderr=log_file, shell=True).returncode != 0:
                print(f'    -> Container failed to build')
                raise RuntimeError(f"Container failed to build\n{' '*14}log @ {docker_log}")
        docker_log.unlink()

    def run_in_container(self, cmd, additional_cmds=[]):
        ## Step 1: build command
        # base within container
        container_name = f'dompteur_{secrets.token_hex(4)}'
        full_cmd = f'docker run '\
            f"--rm  --name {container_name} "\
            f'-v {self.base_dir}:/root/experiment/ '\
            f'-v {self.base_dir.joinpath("exp")}:/root/kaldi/wsj_recipe/exp '\
            f'-v {self.base_dir.joinpath("data")}:/root/kaldi/wsj_recipe/data '\
            f'-v {self.base_dir.joinpath("adversarial_examples")}:/root/kaldi/wsj_recipe/adversarial_examples '\
            f'{" ".join(additional_cmds)} '\
            f'dompteur '\
            f'/bin/bash -c "cd /root/kaldi/wsj_recipe/ && {cmd} && chown -R {os.geteuid()}:{os.geteuid()} /root/experiment"'
        ## Step 2: execute
        start = time.time()
        with open(self.log_file, 'a+') as log_file:
            log_file.write(f'\n{"#"*100}\n{cmd}\n{"#"*100}\n{full_cmd}\n{"#"*100}\n\n')
            try:
                if run(full_cmd, stdout=log_file, stderr=log_file, shell=True).returncode != 0:
                    raise RuntimeError('Container returned statuscode != 0. See `kaldi_log.txt` for more details.')
            except KeyboardInterrupt as e:
                run(f'docker kill {container_name}', stdout=log_file, stderr=log_file, shell=True)
                raise e
        end = time.time()
        running_time = int(end - start)
        return running_time
            
    def fix_permissions(self):
        self.run_in_container('true')
            
    def cleanup(self):
        self.fix_permissions()
        # remove unnecessary files
        to_remove = []
        to_remove += [ self.base_dir.joinpath('adversarial_examples') ]
        to_remove += list([f for f in self.base_dir.glob('*.txt') if f.name != "training.log.txt"])
        to_remove += [ json_file for json_file in self.base_dir.glob('*.json')]
        to_remove += [path for path in self.base_dir.joinpath('exp').glob('*') 
                           if path.name != 'tri4b' and path.name != 'nnet5d_gpu_time']
        # network
        network_dir = self.base_dir.joinpath('exp/nnet5d_gpu_time')
        to_remove += list(network_dir.glob('adversarial_*'))
        to_remove += list(network_dir.glob('decode_*'))
        to_remove += [path for path in network_dir.glob('*.mdl') if not path.stem == 'final']
        to_remove += [ network_dir.joinpath('log'), network_dir.joinpath('egs') ]
        # tri4b
        tri4b_dir = self.base_dir.joinpath('exp/tri4b')
        to_remove += [ tri4b_dir.joinpath('graph_tgpr'), tri4b_dir.joinpath('log')]
        to_remove += [ path for path in tri4b_dir.glob('*.gz')]
        to_remove += [ path for path in tri4b_dir.glob('trans*')]
        to_remove += list(tri4b_dir.glob('decode_*'))
        # data
        to_remove += [path for path in self.base_dir.joinpath('data').glob('*') if not path.name == 'lang']
        for path in to_remove:
            if path.is_dir():
                shutil.rmtree(path)
            if path.is_file():
                path.unlink()
        # replace symlinks with the actual target
        for symlink in [ e for e in self.base_dir.rglob('*') if e.is_symlink() ]:
            real_dst = symlink.resolve()
            symlink.unlink()
            real_dst.rename(symlink)
        # replace symlinks with the actual target
        for symlink in [ e for e in self.base_dir.rglob('*') if e.is_symlink() ]:
            real_dst = symlink.resolve()
            symlink.unlink()
            real_dst.rename(symlink)

    @staticmethod
    def from_trained_model(model_dir, base_dir, cleanup=False):
        # copy files
        if base_dir.is_dir():
            print(f'\n[!] Kaldi instance "{base_dir.name}" already exists')
            return
        else:
            print(f'\n[+] Copy {model_dir.name} to {base_dir.name}')
            shutil.copytree(src=str(model_dir), dst=str(base_dir), symlinks=True)
        # create kaldi instance
        kaldi = Kaldi(base_dir)
        if cleanup:
            print("    -> clean up")
            kaldi.cleanup()
        return kaldi

    def decode_wav(self, wav, phi=None):
        wav = Path(wav)
        with TemporaryDirectory() as data_dir:
            shutil.copyfile(wav, Path(data_dir).joinpath(wav.name))
            _, decoding_meta = self.decode_wavs(data_dir, None, phi) 
        return ' '.join(decoding_meta.pop()['hyp'].split())

    def decode_wavs(self, data_dir, text, phi=None):
        data_dir = Path(data_dir)
        # create decode dir
        decode_name = f'decode_job_{data_dir.name}_{int(time.time())}'
        decode_dir = self.base_dir.joinpath(f"exp/nnet5d_gpu_time/{decode_name}")
        # create kaldi dataset
        dataset = Dataset(data_dir, name=decode_name)
        dataset.text = text
        dataset.dump_as_kaldi_dataset(decode_dir, wavs_prefix=f"exp/nnet5d_gpu_time/{decode_name}")
        # invoke decode script in container
        self.run_in_container(
            f'./decode_wavs.sh exp/nnet5d_gpu_time/{decode_name}',
            additional_cmds=[f'-e NUMJOBS={min(len(dataset), cpu_count())}',
                             f'-e PHI={phi} -e LOG_CLAMP=0']
        )
        # get best_wer
        best_wer = open(f'{decode_dir}/scoring_kaldi/best_wer').read().strip()
        best_wer = float(re.findall(r'%WER (.*) \[', best_wer)[0])
        # get decoding meta
        decoding_meta = parse_per_utt_file(decode_dir)
        return best_wer, decoding_meta
