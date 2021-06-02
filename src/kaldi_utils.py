import json
import re
import sys
import threading
import time
from pathlib import Path

from colorama import Fore, Style
from tqdm import tqdm


def parse_per_utt_file(decode_dir):
    per_utt = Path(decode_dir).joinpath('scoring_kaldi', 'wer_details', 'per_utt').read_text()

    utterances = list(set([ line.split(' ')[0] for line in per_utt.splitlines() ]))

    entries = []
    for utt in utterances:
        ref, hyp, op, csid = [ line for line in per_utt.splitlines() 
                                    if line.startswith(utt)           ]
        
        hyp = hyp.split('hyp')[-1].strip()
        ref = ref.split('ref')[-1].strip()

        N = len(ref.split('ref')[-1].strip().split())
        c, s, i, d = csid.split(' ')[-4:]

        wer = (int(s) + int(i) + int(d))/N

        entries.append({
            'wav_name' : utt,
            'ref' : ref,
            'hyp' : hyp,
            'wer' : wer
        })

    return list(sorted(entries, key=lambda e: e['wer']))

def parse_results_file(base_dir):
    results_files = list(base_dir.joinpath('exp/nnet5d_gpu_time/').glob('*.json'))
    if len(results_files) > 1:
        print('    More than one results file exists')
        return []
    return [f'{float(result.split(" ")[1]):>6.2f}%' for result in json.loads(results_files[0].read_text())]

class KaldiLogger:

    _loggers = []

    def __init__(self, base_dir):
        KaldiLogger._loggers.append(self)
        self.base_dir = base_dir

    @staticmethod
    def stop_all():
        for logger in KaldiLogger._loggers:
            logger.stop_flag = True

    def stop(self):
        self.stop_flag = True
        time.sleep(3)

    @property
    def kaldi_log(self):
        return self.base_dir.joinpath('kaldi_log.txt').read_text()

    @property
    def train_itr(self):
        return len([ l for l in self.kaldi_log.splitlines() if "Training neural net" in l ])

    def log_ae(self, max_inner_itr=50, max_outer_itr=10):
        time.sleep(5)
        self.stop_flag = False
        logger_thread = threading.Thread(target=self._log_ae, args=(max_inner_itr, max_outer_itr))
        logger_thread.start()

    def log_training(self, max_itr=920):
        self.stop_flag = False
        logger_thread = threading.Thread(target=self._log_training, args=(max_itr,))
        logger_thread.start()

    def wait_for_log_entry(self, entry):
        start = time.time()
        while not self.stop_flag:
            if entry in self.kaldi_log:
                break
            time.sleep(1)
        running_time = int(time.time() - start)
        print(f"       completed in {running_time // 3600}h {(running_time % 3600) // 60}m {(running_time % 60)}s")

    def _log_training(self, max_itr):
        time.sleep(10)
        print(f'    -> data preparation I')
        self.wait_for_log_entry("[+] convert training data")
        print(f'    -> convert training data')
        self.wait_for_log_entry("[+] data preparation II")
        print(f'    -> data preparation II')
        self.wait_for_log_entry("[+] train acoustic models")
        print(f'    -> train acoustic models')
        self.wait_for_log_entry("[+] train model")
        print(f'    -> train model')
        with tqdm(total=max_itr, bar_format='       {l_bar}{bar:30}{r_bar}') as pbar:
            pbar.update(self.train_itr)
            while not self.stop_flag:

                if not self.base_dir.is_dir():
                    pbar.set_description(f'Could not find experiment {exp_name.upper()}')
                    time.sleep(1)
                    continue

                if pbar.n < self.train_itr:
                    pbar.update(1)

                time.sleep(1)

        with tqdm(total=max_itr, bar_format='       {l_bar}{bar:30}{r_bar}') as pbar:
            pbar.update(self.train_itr)
            while not self.stop_flag:

                if not self.base_dir.is_dir():
                    pbar.set_description(f'Could not find experiment {exp_name.upper()}')
                    time.sleep(1)
                    continue

                if pbar.n < self.train_itr:
                    pbar.update(1)

                time.sleep(1)

    def _log_ae(self, max_inner_itr=50, max_outer_itr=10):
        self.current_itr = 0
        self.tic = time.time()
        self.running_time = 0
        while not self.stop_flag:
            no_of_lines = self._print_ae_status(max_inner_itr, max_outer_itr)
            time.sleep(1)
            erase_line(no_of_lines)
            self._ae_decoding()
    
    def _print_ae_status(self, max_inner_itr, max_outer_itr):
        no_of_lines = 3
        print(f'\n    {self.base_dir.name.upper()}\n')

        # WERs
        try:
            results_files = list(self.base_dir.joinpath('exp/nnet5d_gpu_time/').glob('*.json'))
            if len(results_files) > 1:
                print('    More than one results file exists')
                return no_of_lines + 1
            results = [float(result.split(' ')[1]) for result in json.loads(results_files[0].read_text())]
            min_wer, min_wer_idx = min(results), results.index(min(results)) + 1
            last_results = [f'{results[idx]:>6.2f}%' for idx in range(max(len(results)-5,0), len(results))]
            current_itr = len(results)
            print(f'    ITERATION {current_itr} / {max_outer_itr}')
            print(f'    -> CURRENT: {results[-1]:>6.2f}% @ {len(results)} ')
            print(f'    -> BEST   : {min_wer:>6.2f}% @ {min_wer_idx}')
            print(f'    -> LAST   : {" -> ".join(last_results)}\n')
            no_of_lines += 5
        except:
            pass

        # finished AEs
        try:
            adversarial_log_dir = [e for e in self.base_dir.joinpath('exp/nnet5d_gpu_time/').glob('adversarial*') if e.is_dir()][0]
            utt_itr = len(adversarial_log_dir.joinpath('scoring_kaldi/wer_details/utt_itr').read_text().splitlines())
            print(f"    FINISHED AEs: {utt_itr}\n")
            no_of_lines += 2
        except:
            pass

        # kaldi log
        try:
            print('    KALDI LOG'); no_of_lines += 1
            kaldi_log = self.base_dir.joinpath('kaldi_log.txt').read_text().splitlines()
            for idx in range(-3, 0):
                l = kaldi_log[idx].strip()
                if len(l) > 50:
                    print(f'    {l[:25]} ... {l[-25:]}')
                else:
                    print(f'    {l}')
                no_of_lines += 1
            print(); no_of_lines += 1
        except:
            pass

        # progress bars
        try:
            adversarial_log_dir = [e for e in self.base_dir.joinpath('exp/nnet5d_gpu_time/').glob('adversarial*') if e.is_dir()][0]
            adversarial_log_files = sorted(adversarial_log_dir.joinpath('log').glob('adversarial.*.log'))
            itrs = []
            if len(adversarial_log_files) > 0:
                in_progress = []
                for adversarial_log_file in adversarial_log_files:
                    adversarial_log = adversarial_log_file.read_text().splitlines()
                    itr = len([l for l in adversarial_log if "SPOOF_ITERATION" in l ]) 
                    
                    uttr_name = re.findall("LOG \(nnet-spoof-iter\[5\.5\]:DecodableAmNnetSpoofIter\(\):nnet2\/decodable-am-nnet\.h:517\) (.*)", "\n".join(adversarial_log))
                    if len(uttr_name) > 0:
                        uttr_name = uttr_name[0]
                    else:
                        uttr_name = ""                

                    if "Ended" not in adversarial_log[-1]:
                        in_progress.append((uttr_name, itr))
                        itrs.append(itr)

                no_of_pbars = 3
                for uttr_name, itr in in_progress[:no_of_pbars]:
                    print(f'    {uttr_name:>8} [{itr:02}/{max_inner_itr} {"#"*(itr//2)+" "*((max_inner_itr-itr)//2)}]')    
                    no_of_lines += 1

                if len(in_progress) > no_of_pbars:
                    print(f"    ... {len(in_progress)-no_of_pbars} more")
                    no_of_lines += 1        

                if self.current_itr < min(itrs):
                    self.running_time = int(time.time() - self.tic)
                    self.tic = time.time()
                    self.current_itr = min(itrs)

                if min(itrs) == 0:
                    self.current_itr = 0
                
                print(f'\n         last     {(self.running_time % 3600) // 60:>2}m {(self.running_time % 60):>2}s')
                remaining = self.running_time * (max_inner_itr - self.current_itr)
                print(f'    remaining {remaining // 3600:>2}h {(remaining % 3600) // 60:>2}m {(remaining % 60):>2}s')
                no_of_lines += 3
               
        except:
            pass

        return no_of_lines

    def _ae_decoding(self):
        decoding_start = "[+] Start decoding AEs"
        decoding_end = "[+] End decoding AEs"
        kaldi_log = self.base_dir.joinpath('kaldi_log.txt').read_text()
        print()
        while kaldi_log.count(decoding_start) > kaldi_log.count(decoding_end):
            print(Fore.RED + "    DECODING" + Style.RESET_ALL)
            time.sleep(0.5); erase_line(1)
            print("            ")
            time.sleep(0.5); erase_line(1)
            kaldi_log = self.base_dir.joinpath('kaldi_log.txt').read_text()
        erase_line(1)

def erase_line(n=1):
    for _ in range(0, n):
        print("\033[F", end="")
        print("\033[K", end="")
