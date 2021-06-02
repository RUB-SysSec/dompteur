#!/bin/bash

. ./cmd.sh 
. ./path.sh

rm -rf exp/nnet5d_gpu_time

steps/nnet2/train_pnorm_fast_time.sh --stage -10 \
   --samples-per-iter 400000 \
   --parallel-opts "-gpu 1" \
   --num-threads "1" \
   --minibatch-size "512" \
   --num-jobs-nnet 1  --mix-up 8000 \
   --initial-learning-rate 0.02 --final-learning-rate 0.004 \
   --num-hidden-layers 4 \
   --pnorm-input-dim 2000 --pnorm-output-dim 400 \
   --cmd "$decode_cmd" \
    data/train_si284 data/lang exp/tri4b exp/nnet5d_gpu_time
