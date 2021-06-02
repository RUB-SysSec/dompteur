# Dompteur: Taming Audio Adversarial Examples

<p>
<img align="right" width="350"  src="media/dompefant.png"> 
</p>

This is the code repository accompaning our USENIX Security '21 paper [Dompteur: Taming Audio Adversarial Examples](https://arxiv.org/abs/2102.05431).

>Adversarial examples seem to be inevitable. These specifically crafted inputs allow attackers to arbitrarily manipulate machine learning systems. Even worse, they often seem harmless to human observers. In our digital society, this poses a significant threat. For example, Automatic Speech Recognition (ASR) systems, which serve as hands-free interfaces to many kinds of systems, can be attacked with inputs incomprehensible for human listeners. The research community has unsuccessfully tried several approaches to tackle this problem.
>
> In this paper we propose a different perspective: We accept the presence of adversarial examples against ASR systems, but we require them to be perceivable by human listeners. By applying the principles of psychoacoustics, we can remove semantically irrelevant information from the ASR input and train a model that resembles human perception more closely. We implement our idea in a tool named DOMPTEUR and demonstrate that our augmented system, in contrast to an unmodified baseline, successfully focuses on perceptible ranges of the input signal. This change forces adversarial examples into the audible range, while using minimal computational overhead and preserving benign performance. To evaluate our approach, we construct an adaptive attacker, which actively tries to avoid our augmentations and demonstrate that adversarial examples from this attacker remain clearly perceivable. Finally, we substantiate our claims by performing a hearing test with crowd-sourced human listeners.

## Audio Samples

As an example for our countermeasure, we uploaded the audio files used for our user study. These samples can be found at [https://rub-syssec.github.io/dompteur/](https://rub-syssec.github.io/dompteur/).

## Models and Datasets

Pre-trained models from our experiments are uploaded [here](https://drive.google.com/drive/folders/1MA8e_NRaOycCd9EgHKIFiT5MhVa0zeQO?usp=sharing). These models are trained on the Wall Street Journal (WSJ) speech corpus. For details on how to train your own models see Section “Train your own models”.

For testing, we also included a small dataset with speech samples (`datasets/speech_10`). If you want to create your own datasets, follow the file structure in the example test set and use audio files that are sampled with 16kHz.

## Prerequisites

We implemented DOMPTEUR based on [Kaldi ASR](https://kaldi-asr.org) and created a Dockerfile with all necessary tools to train the system and run the attacks. It can be build via

```
git clone git@github.com:RUB-SysSec/dompteur.git ~/dompteur
cd ~/dompteur; docker build -t dompteur .
```

Moreover, for our experiments we prepared various convenience scripts to run commands within this container. To use these scripts, you need a recent version of python and install the requirements (i.e., `pip3 install -r requirements.txt`). We recommend a virtual environment for this.


## Decoding

We included a small dataset with speech samples (@ `datasets/speech_10`). To check if everything is setup correctly, you can download a pre-trained model and decode this dataset.

The syntax of `decode.py` script is as follows:

```
usage: decode.py [-h] [--models MODELS] [--experiments EXPERIMENTS]
                 [--dataset_dir DATASET_DIR] [--phi PHI] [--low LOW]
                 [--high HIGH]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS       Directory with trained models.
  --experiments EXPERIMENTS
                        Output directory for experiments.
  --dataset_dir DATASET_DIR
                        Path to dataset.
  --phi PHI             Scaling factor for the psychoacoustic filter.
  --low LOW             Lower cut-off frequency of band-pass filter.
  --high HIGH           Higher cut-off frequency of band-pass filter.
```

With the standard model (`dompteur_f825_phi.None_bandpass.None-None`), you should see a WER of 4.88% for the test dataset.

## Adversarial Examples

Similar, you can use the `attack.py` script to compute adversarial examples: 

```
usage: attack.py [-h] [--models MODELS] [--experiments EXPERIMENTS]
                 [--dataset_dir DATASET_DIR] [--inner_itr INNER_ITR]
                 [--max_itr MAX_ITR] [--learning_rate LEARNING_RATE]
                 [--phi PHI] [--low LOW] [--high HIGH]
                 [--attacker {baseline,adaptive}]
                 [--psycho_hiding_thresh PSYCHO_HIDING_THRESH]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS       Directory with trained models.
  --experiments EXPERIMENTS
                        Output directory for experiments.
  --dataset_dir DATASET_DIR
                        Path to target dataset.
  --inner_itr INNER_ITR
                        Number of optimization steps in inner loop. After
                        <inner_itr> steps, AEs are deocded and hearing
                        thresholds refreshed.
  --max_itr MAX_ITR     Maximum number of optimization steps.
  --learning_rate LEARNING_RATE
                        Learning rate for the attack.
  --phi PHI             Scaling factor for the psychoacoustic filter.
  --low LOW             Lower cut-off frequency of band-pass filter.
  --high HIGH           Higher cut-off frequency of band-pass filter.
  --attacker {baseline,adaptive}
                        Type of attacker.
  --psycho_hiding_thresh PSYCHO_HIDING_THRESH
                        Margin "lambda" in dB for psychoacoustic hiding.
                        Disabled for -1.
```

## Train your own models

In case you want to train your own models, you can use `train.py` to train your system on the Wall Street Journal (WSJ) speech corpus:

```
usage: train.py [-h] [--models MODELS] [--wsj_dir WSJ_DIR] [--phi PHI]
                [--low LOW] [--high HIGH] [--gpus GPUS]

optional arguments:
  -h, --help         show this help message and exit
  --models MODELS    Directory for trained models.
  --wsj_dir WSJ_DIR  Path to Wall Street Journal (WSJ) specch corpus.
  --phi PHI          Scaling factor for the psychoacoustic filter.
  --low LOW          Lower cut-off frequency of band-pass filter.
  --high HIGH        Higher cut-off frequency of band-pass filter.
  --gpus GPUS        GPU devices used for training.
```

## Psychoacoustic Filtering

Finally, if you only want to use the psychoacoustic filter (e.g., for a different application) we included an additional script. This script takes as input a directory with audio files and applies the psychoacoustic filtering on all input files.

```
usage: psychoacoustic.py [-h] --data_in_dir DATA_IN_DIR --data_out_dir
                         DATA_OUT_DIR [--phi PHI]

optional arguments:
  -h, --help            show this help message and exit
  --data_in_dir DATA_IN_DIR
                        Directory with wavs.
  --data_out_dir DATA_OUT_DIR
                        Output directory for processed wavs.
  --phi PHI             Scaling factor for the psychoacoustic filter.
```

More details on the conversion can be found in `hearing_thresholds` and `kaldi/wsj_recipe/psycho/psycho.py`.