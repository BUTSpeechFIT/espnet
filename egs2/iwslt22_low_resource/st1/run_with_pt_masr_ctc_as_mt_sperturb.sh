#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $# -ne 9 ]; then
    echo ""
    echo "$0 <nbpe_from_pretrained_ASR> <pretrained_bpedir> <st_config.yaml> <decode_config.yaml> <st_exp/> <pretrained_asr> <pfx> <stage> <stop_stage>"
    echo " - Require 9 args. Given $#"
    echo """
  stage   10: collect stats - specific for every train/val/test sets
  stage   11: train model
  stage   12: decode
  stage   13: scoring
"""
    exit;
fi

# language related
lang_pair="taq-fr"
tgt_lang="fr"
src_lang="fr" # since we want to use CTC directly for translation

# adding speed perturb data for fine-tuning helps
speed_pertub_factors="0.9 1.0 1.1"

# src_nbpe=$1
tgt_nbpe=$1
src_case=tc
tgt_case=tc
token_joint=true

feats_type=fbank_pitch

inference_st_model="valid.acc.ave.pth"

datadir="data"

train_set="train"
dev_set="valid"
test_set="valid test"


token_listdir=$2
st_config=${3}
inference_config=$4

st_exp=$5
mkdir -p "${st_exp}"

pretrained_asr=$6
pfx=$7
stage=$8
stop_stage=$9

st_stats_dir=st_stats/${pfx}_asr_init_${lang_pair}_${tgt_case}_unigram_${tgt_nbpe}_ctc_as_mt
mkdir -p "${st_stats_dir}"

dump_dir=dump
# mkdir -p ${dump_dir}

if [ "${stage}" -lt 5 ] && [ "${stop_stage}" -gt 5 ]; then
    echo "Only stages 2, 3, 4 and 10, 11, 12, 13, .. are allowed."
    echo "You should not train BPE, instead you should use the one from pre-trained ASR model."
    exit;
fi

./st.sh \
    --speed_perturb_factors "${speed_pertub_factors}" \
    --ngpu 1 \
    --use_lm false \
    --pretrained_asr "${pretrained_asr}" \
    --feats_type ${feats_type} \
    --token_joint ${token_joint} \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --datadir ${datadir} \
    --dumpdir ${dump_dir} \
    --st_stats_dir ${st_stats_dir} \
    --st_exp ${st_exp} \
    --src_lang ${src_lang} \
    --src_case ${src_case} \
    --tgt_lang ${tgt_lang} \
    --tgt_token_type "bpe" \
    --src_token_type "bpe" \
    --src_bpedir ${token_listdir} \
    --tgt_bpedir ${token_listdir} \
    --src_nbpe ${tgt_nbpe} \
    --tgt_nbpe ${tgt_nbpe} \
    --tgt_case ${tgt_case} \
    --token_listdir ${token_listdir} \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${dev_set}" \
    --test_sets "${test_set}" \
    --tgt_bpe_train_text "${datadir}/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --src_bpe_train_text "${datadir}/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --inference_st_model ${inference_st_model} \
    --copy_feats_to_dir "/tmp/" \
    --utt_extra_files "text.${tgt_case}.${tgt_lang}" \
    --multilingual_mode true \
    --input_token_list_ftype "token_flist" \
    --lid ${tgt_lang}
