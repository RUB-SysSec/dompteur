import re
from pathlib import Path


def select_model(models, phi, low, high, model_idx=0):
    # get all models for parameters
    model_dirs = [ model_dir for model_dir in models.glob(f'*phi.{phi}_bandpass.{low}-{high}')
                             if model_dir.joinpath('training.log.txt').is_file()                ]
    # get WER for each model
    wers = []
    for model_dir in model_dirs:
        wer = [float(re.findall(r'%WER (.*) \[', l)[0]) for l in model_dir.joinpath('training.log.txt').read_text().splitlines() if l.startswith('%WER')]
        assert len(wer) == 1
        wer = wer.pop()
        wers.append((model_dir, wer))
    # assert that we found at least one model
    assert len(wers) != 0
    wers = sorted(wers, key=lambda x: x[1])
    # return the i'th model
    model, _ = wers[model_idx]
    return model
