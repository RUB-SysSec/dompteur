import json
import random
import shutil
from collections import OrderedDict
from pathlib import Path


class Dataset:

    def __init__(self, data_dir, name=None):
        self.data_dir = Path(data_dir)
        self.name = name if name else self.data_dir.name 
        self.wavs = sorted([wav.stem for wav in self.data_dir.glob("*.wav")
                                     if not wav.stem.startswith('._')])
        # check if texts are available
        text_file = self.data_dir.joinpath('text.json')
        self.text = json.loads(text_file.read_text()) if text_file.is_file() else {}
        # check if targets are available
        target_file = self.data_dir.joinpath('target.json')
        self.target = json.loads(target_file.read_text()) if target_file.is_file() else {}     

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.wavs)

    @staticmethod
    def init_dataset(dataset_dir, wav_dir, wavs, text, target):
        dataset_dir = Path(dataset_dir)
        if dataset_dir.is_dir():
            print(f"[!] {dataset_dir} already exists")
        dataset_dir.mkdir()
        for wav in wavs:
            shutil.copyfile(src=Path(wav_dir, f'{wav}.wav'),
                            dst=Path(dataset_dir, f'{wav}.wav'))
            assert wav in text, f'{wav} not in text'
            assert wav in target, f'{wav} not in target'
        Path(dataset_dir, 'target.json').write_text(json.dumps(target, indent=4))
        Path(dataset_dir, 'text.json').write_text(json.dumps(text, indent=4))

    def dump_as_kaldi_dataset(self, out_dir, wavs_prefix, target_idx=None):
        out_dir = Path(out_dir); out_dir.mkdir()
        # copy data
        shutil.copytree(self.data_dir, out_dir.joinpath('wavs'))
        # dump targets
        target_utterances = list(set(self.target.values()))
        targets_dir = out_dir.joinpath('target_utterances')
        targets_dir.mkdir()
        targets_dir.joinpath('target-utterances.txt').write_text("\n".join(target_utterances) + "\n")
        # create meta info
        wavs = OrderedDict()
        spk2utt = OrderedDict()
        spk2gender = OrderedDict()
        text = OrderedDict()
        target = OrderedDict()
        for idx, utt in enumerate(self.wavs):
            spk_id = 'spk{:0>8}'.format(idx)
            spk2utt[spk_id] = [utt]
            spk2gender[spk_id] = random.choice(['m', 'f'])
            text[utt] = self.text[utt] if self.text else "DATA WITH NO SPOKEN CONTENT"
            target[utt] = target_utterances.index(self.target[utt]) if self.target else "-1"
        # dump to disk
        # -> spk2utt: spk utt1 utt2 ...
        spk2utt_out = [f'{spk} {" ".join(utt)}' for spk, utt in spk2utt.items()]
        Path(f'{out_dir}/spk2utt').write_text("\n".join(spk2utt_out)+"\n")
        # -> spk2gender: spk [m/f]
        spk2gender_out = [f'{spk} {gender}' for spk, gender in spk2gender.items()]
        Path(f'{out_dir}/spk2gender').write_text("\n".join(spk2gender_out)+"\n")
        # -> target: utt target_idx
        target_out = [f'{utt} {target_id}'  for utt, target_id in target.items()]
        Path(f'{out_dir}/target').write_text("\n".join(target_out)+"\n")
        # -> text: utt transcription
        text_out = [f'{utt} {text}' for utt, text in text.items()]
        Path(f'{out_dir}/text').write_text("\n".join(text_out)+"\n")
        # -> wav.scp: utt path
        wav_out = [f'{utt} {wavs_prefix}/wavs/{utt}.wav' for utt in self.wavs]
        Path(f'{out_dir}/wav.scp').write_text("\n".join(wav_out)+"\n")
