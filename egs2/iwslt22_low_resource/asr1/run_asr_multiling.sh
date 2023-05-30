#!/bin/bash

set -e
set -u
set -o pipefail

if [ $# -lt 5 ]; then
    echo "$0 <lang_suffix> <nbpe> <train_config.yaml> <output_exp_dir> <collect_stats or train or decode> [lid for decoding: fr.tc or fr] [token_listdir for scoring]"
    echo "Require atleast 5 args. Given $#"
    echo "Eg: $0 6L.300.tc 1000 conf/train_asr_ctc0.3.yaml  exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/ collect_stats "
    echo "Eg: $0 6L.300.tc 1000 conf/train_asr_ctc0.3.yaml  exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/ train "
    echo "Eg: $0 6L.300.tc 1000 conf/train_asr_ctc0.3.yaml  exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/ decode de.tc "
    exit;
fi


PRE=$PWD

lang_sfx=$1

token_type="bpe"
nbpe=$2

# change the following three dirs appropriately
dumpdir=${PRE}/dump
datadir=${PRE}/data_subset
token_listdir=${PRE}/token_list/${lang_sfx}

echo "- data dir: ${datadir}"
echo "- dump dir: ${dumpdir}"
echo "- token listdir: ${token_listdir}"

use_lm=false

feats_type="fbank_pitch"

inference_nj=32
input_size=83  # fbank+pitch


asr_config=$3
inference_config=conf/decode.yaml

train_set="train_${lang_sfx}"
valid_set="dev_${lang_sfx}"

output_dir=$4
mkdir -pv ${output_dir}

step=$5

lang_case=${6:-""}
lid=$(echo ${lang_case} | cut -d'.' -f1)
test_sets="dev_${lang_case} test_${lang_case}"

if [ ! -f ${dumpdir}/${feats_type}/${train_set}/feats.scp ]; then
    echo "File ${dumpdir}/${feats_type}/${train_set}/feats.scp not found."
    exit;
fi

if [ ! -f ${dumpdir}/${feats_type}/${valid_set}/feats.scp ]; then
    echo "File ${dumpdir}/${feats_type}/${valid_set}/feats.scp not found."
    exit;
fi

token_flist=${token_listdir}/bpe_unigram${nbpe}/tokens.txt
bpe_flist=${token_listdir}/bpe_unigram${nbpe}/bpe.model

if [ ! -f ${token_flist} ]; then
    echo ${token_flist}" not found."
    exit;
fi

stats_dir=asr_stats/multi_${lang_sfx}_${token_type}_${nbpe}
mkdir -pv ${stats_dir}

if [ "${step}" == "collect_stats" ]; then

    # works
    ./asr.sh \
        --stage 10 \
        --stop_stage 10 \
        --feats_type ${feats_type} \
        --token_listdir ${token_listdir} \
        --dumpdir ${dumpdir} \
        --token_type ${token_type} \
        --nbpe ${nbpe} \
        --use_lm ${use_lm} \
        --asr_config "${asr_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --asr_stats_dir "${stats_dir}" \
        --asr_exp "${output_dir}" \
        --multilingual_mode true \
        --input_token_list_ftype "token_flist"


elif [ "${step}" == "train" ]; then

    # works
    ./asr.sh \
        --stage 11 \
        --stop_stage 11 \
        --feats_type ${feats_type} \
        --token_listdir ${token_listdir} \
        --dumpdir ${dumpdir} \
        --token_type ${token_type} \
        --nbpe ${nbpe} \
        --use_lm ${use_lm} \
        --asr_config "${asr_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --asr_stats_dir "${stats_dir}" \
        --asr_exp "${output_dir}" \
        --multilingual_mode true \
        --copy_feats_to_dir "/tmp/" \
        --input_token_list_ftype "token_flist" \
        --asr_args "--recopy"

elif [ "${step}" == "decode" ]; then

    if [ "${lid}" == "" ]; then
        echo "${lid} is requried for decoding"
        exit;
    fi

    for dec_set in ${test_sets}; do
        if [ ! -f ${dumpdir}/${feats_type}/${dec_set}/feats.scp ]; then
            echo "File ${dumpdir}/${feats_type}/${dec_set}/feats.scp not found."
            exit;
        fi
    done

    ./asr.sh \
        --stage 12 \
        --stop_stage 12 \
        --feats_type ${feats_type} \
        --token_listdir ${token_listdir} \
        --dumpdir ${dumpdir} \
        --token_type ${token_type} \
        --nbpe ${nbpe} \
        --use_lm ${use_lm} \
        --asr_config "${asr_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --asr_stats_dir "${stats_dir}" \
        --asr_exp "${output_dir}" \
        --inference_config "${inference_config}" \
        --inference_nj ${inference_nj} \
        --multilingual_mode true \
        --lid ${lid} \

elif [ "${step}" == "score" ]; then

    if [ ! -z $7 ]; then
        token_listdir=$7
    fi
    ./asr.sh \
        --stage 13 \
        --stop_stage 13 \
        --feats_type ${feats_type} \
        --token_listdir ${token_listdir} \
        --dumpdir ${dumpdir} \
        --token_type ${token_type} \
        --nbpe ${nbpe} \
        --use_lm ${use_lm} \
        --asr_config "${asr_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --asr_stats_dir "${stats_dir}" \
        --asr_exp "${output_dir}" \
        --inference_config "${inference_config}" \
        --multilingual_mode true \
        --lid ${lid}

fi
