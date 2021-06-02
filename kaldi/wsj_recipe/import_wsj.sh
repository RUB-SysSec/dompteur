#!/bin/bash

. ./cmd.sh 
. ./path.sh

wsj0=/root/WSJ/WSJ0/csr_senn_d?
wsj1=/root/WSJ/WSJ1/csr_senn_d?
local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
local/wsj_prepare_dict.sh --dict-suffix "_nosp"
utils/prepare_lang.sh data/local/dict_nosp "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp
local/wsj_format_data.sh --lang-suffix "_nosp"