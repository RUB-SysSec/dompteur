
. ./cmd.sh 
. ./path.sh

decode_dir=$1

# preprocess wavs
python3 psycho/pre-processing.py --encoding_dir ${decode_dir}/wavs

# format dataset
utils/spk2utt_to_utt2spk.pl ${decode_dir}/spk2utt > ${decode_dir}/utt2spk
utils/fix_data_dir.sh ${decode_dir}

# make timing features
steps/make_time.sh --cmd "$train_cmd" --nj ${NUMJOBS} ${decode_dir}
steps/compute_cmvn_stats.sh ${decode_dir}

# decode dataset
steps/nnet2/decode.sh --cmd "$decode_cmd" --nj ${NUMJOBS} \
    exp/tri4b/graph_bd_tgpr ${decode_dir} ${decode_dir}

# print results
cat ${decode_dir}/scoring_kaldi/best_wer