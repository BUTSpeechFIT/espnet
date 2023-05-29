#!/bin/bash

set -e

nargs=7
if [ $# -lt ${nargs} ]; then
    echo "$0 <asr_expdir> <lm_exp> <ctc_wgt> <lm_wgt> <n-best> <lang> <multilingual_mode: true or false> [nj: 32]"
    echo "- Joint decoding with ctc weight lm weight and N-best decoding."
    echo "- Require ${nargs}. Given $#"
    exit;
fi

datadir="data_v3"
dumpdir="dump_v3"

lang=${6}
mul_mode=${7}

train_set="train_${lang}"
dev_set="dev_${lang}"
test_set="dev_${lang} test_${lang}"

asr_exp=${1}
lm_exp=${2}
ctc_wgt=${3}
lm_wgt=${4}

nbest=${5}
nj=${8:-32}

asr_config=${asr_exp}/config.yaml
if [ ! -f ${asr_config} ]; then
    echo "${asr_config} FILE NOT FOUND."
    exit;
fi
bpemodel=$(grep -E "^bpemodel:" "${asr_config}" | awk -F": " '{print $NF}')
bpedir=$(dirname ${bpemodel})
nbpe=$(wc -l ${bpedir}/tokens.txt | cut -d' ' -f1)
echo "nbpe: ${nbpe}"

use_lm=true

./asr.sh \
    --ngpu 0 \
    --lang ${lang} \
    --inference_nj ${nj} \
    --use_lm ${use_lm} \
    --feats_type fbank_pitch \
    --stage 12 \
    --stop_stage 13 \
    --datadir ${datadir} \
    --dumpdir ${dumpdir} \
    --asr_exp ${asr_exp} \
    --lm_exp ${lm_exp} \
    --nbpe ${nbpe} \
    --token_listdir ${bpedir}/../ \
    --asr_config ${asr_config} \
    --inference_config conf/decode.yaml \
    --train_set ${train_set} \
    --valid_set ${dev_set} \
    --test_sets "${test_set}" \
    --inference_asr_model valid.acc.ave.pth \
    --inference_tag "decode_asr_ctc_${ctc_wgt}_lm_${lm_wgt}_model_valid.acc.ave" \
    --inference_args "--ctc_weight ${ctc_wgt} --lm_weight ${lm_wgt} --nbest ${nbest}" \
    --multilingual_mode ${mul_mode} \
    --lid ${lang}
done
