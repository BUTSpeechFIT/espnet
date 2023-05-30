#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $# -ne 7 ]; then
    echo "$0 <lang or lang_with_sfx> <asr_config> <nbpe> <token_listdir> <asr_exp_dir> <stage> <stop_stage>"
    echo "eg: $0 hi conf/train_asr.yaml 1000 token_listdir/hi/ exp/transformer/hi_1000bpe_12L_6L_256d_0.3ctc_0.1d_0.0005lr_100e/ 2 11"
    echo " Require 7 args. Given $#"
    echo " Stage 2: speed perturb"
    echo " Stage 3: fbank / feature extraction"
    echo " Stage 4: remove long / short utt"
    echo " Stage 5: tokenizer"
    echo " Stage 6-9: LM"
    echo " Stage 10: collect stats"
    echo " Stage 11-13: train, decode, score"
    exit;
fi

# change the following appropriately
datadir=data_v3
dumpdir=dump_v3

echo "- data dir: ${datadir}"
echo "- dump dir: ${dumpdir}"

nj=32

lang=$1

echo "- lang  : ${lang}"

train_set=train_"${lang}"
valid_set=dev_"${lang}"
test_set="dev_${lang} test_${lang}"

speed_perturb_factors="1.0"

asr_config=${2}
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=false
token_type="bpe"
nbpe=${3}

mkdir -pv $4
token_listdir=$4

mkdir -p $5
asr_exp_dir=$5

stage=$6
stop_stage=$7

remove_dups=1

asr_stats_dir="asr_stats_v2/${lang}_${token_type}_${nbpe}"
mkdir -pv ${asr_stats_dir}

./asr.sh \
    --remove_dups ${remove_dups} \
    --nj ${nj} \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --ngpu 1 \
    --lang "${lang}" \
    --datadir ${datadir} \
    --dumpdir ${dumpdir} \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
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
