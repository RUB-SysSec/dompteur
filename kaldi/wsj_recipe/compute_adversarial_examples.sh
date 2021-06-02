#!/bin/bash

. ./cmd.sh 
. ./path.sh

dataset=$1
thresh=$2
itr=$3
max_itr=$4
numjobs=$5
split=$6
attacker_type=$7

echo Parsed commandline params: dataset=${dataset} thresh=${thresh} \
                                itr=${itr} max_itr=${max_itr} numjobs=${numjobs} \
                                split=${split} attacker_type=${attacker_type}

dataset_dir=data/${dataset}
dataset_decode_dir=exp/nnet5d_gpu_time/decode_${dataset}
adversarial_dir=exp/nnet5d_gpu_time/adversarial_${dataset}
adversarial_data_dir=data/adversarial_${dataset}

echo "[+] Computing adversarial examples"

## Step 1: preprocess dataset

# fix dataset
utils/fix_data_dir.sh ${dataset_dir}
utils/spk2utt_to_utt2spk.pl ${dataset_dir}/spk2utt > ${dataset_dir}/utt2spk

# make timing features
mkdir -p ${dataset_decode_dir}/utterances
steps/make_time.sh --cmd "$train_cmd" --nj $split ${dataset_dir} || exit 1;
steps/compute_cmvn_stats.sh ${dataset_dir} || exit 1;

# decode dataset
steps/nnet2/decode.sh --cmd "$decode_cmd" --nj $numjobs \
    exp/tri4b/graph_bd_tgpr ${dataset_dir} ${dataset_decode_dir}

# move utterances to adversarial folder
mkdir -p ${adversarial_dir}
mv ${dataset_decode_dir}/utterances ${adversarial_dir}


## Step 2: compute hearing thresholds
if [ $thresh != -1 ]; then
    mkdir -p ${adversarial_dir}/thresholds
    /root/hearing_thresholds/run_calc_threshold.sh /usr/local/MATLAB/MATLAB_Runtime/v96 ${dataset_dir}/wav.scp 512 256 ${adversarial_dir}/thresholds/
else
    echo "skip hearing thresholds"  
fi

## Step 3: compute adversarial examples
steps/nnet2/adversarial/adversarial_mt.sh --cmd "$decode_cmd" --nj $numjobs --thresh $thresh --numiter $itr --maxitr $max_itr --experiment ${dataset} --attacker_type ${attacker_type} \
    exp/tri4b/graph_bd_tgpr ${dataset_dir} ${adversarial_dir} exp/tri4b data/lang exp/nnet5d_gpu_time/tree ${adversarial_data_dir} nnet5d_gpu_time exp/nnet5d_gpu_time
