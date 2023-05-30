#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $# -lt 8 ]; then
    echo "$0 <datadir> <lang or lang_with_sfx> <asr_config> <nbpe> <token_listdir> <asr_exp_dir> <stage> <stop_stage> [ngpu: 1]"
    echo "eg: $0 data_subset fr.tc conf/train_asr.yaml 1000 token_listdir/fr.50.tc/ exp.tc/transformer/fr.50.tc_1000bpe_12L_6L_256d_0.3ctc_0.1d_0.0005lr_100e/ 11 11"
    echo " Require 8 args. Given $#"
    echo " Stage 3: fbank / feature extraction"
    echo " Stage 4: remove long / short utt"
    echo " Stage 5: tokenizer"
    echo " Stage 6-9: LM"
    echo " Stage 10: collect stats"
    echo " Stage 11-13: train, decode, score"
    exit;
fi

nj=20

datadir=${1}

lang_sfx=${2}
lang=$(echo ${lang_sfx} | cut -d'.' -f1)

echo $lang_sfx
echo $lang

train_set=train_"$(echo "${lang_sfx}" | tr - _)"
valid_set=dev_"$(echo "${lang_sfx}" | tr - _)"
test_set="${valid_set} test_$(echo ${lang_sfx} | tr - _)"

asr_config=${3}
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=false
token_type="bpe"
nbpe=${4}

mkdir -p $5
token_listdir=$5

mkdir -p $6
asr_exp_dir=$6

stage=$7
stop_stage=$8

asr_stats_dir="asr_stats/${lang_sfx}_${token_type}_${nbpe}"
mkdir -pv ${asr_stats_dir}

ngpu=${8:-1}

./asr.sh \
    --nj ${nj} \
    --ngpu ${ngpu} \
    --lang "${lang}" \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --datadir ${datadir} \
    --use_lm ${use_lm} \
    --lm_config "${lm_config}" \
    --token_type ${token_type} \
    --nbpe $nbpe \
    --token_listdir ${token_listdir} \
    --feats_type fbank_pitch \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --bpe_train_text "${datadir}/${train_set}/text" \
    --lm_train_text "${datadir}/${train_set}/text" \
    --asr_exp ${asr_exp_dir} \
    --asr_stats_dir ${asr_stats_dir} \
    --copy_feats_to_dir "/tmp/"
