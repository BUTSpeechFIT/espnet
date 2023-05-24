#!/bin/bash

set -e

nargs=2
if [ $# -lt ${nargs} ]; then
    echo "$0 <st_exp> <n-best> [nj: 32]"
    echo "- N-best decoding. This is for multilingual model"
    echo "- Require ${nargs}. Given $#"
    exit;
fi

train_set='train'
dev_set='dev'
test_set="test"

st_exp=$1
nbest=${2}
nj=${3:-32}
st_config=${st_exp}/config.yaml

if [ ! -f ${st_config} ]; then
    echo "${st_config} FILE NOT FOUND."
    exit;
fi

nbpe=200  # this option does not matter
tgt_lang=hi

bpemodel=$(grep -E "^bpemodel:" "${st_config}" | awk -F": " '{print $NF}')
bpedir=$(dirname ${bpemodel})


for d in 0.3 ; do
    ./st.sh \
        --ngpu 0 \
        --inference_nj ${nj} \
        --use_lm false \
        --feats_type fbank_pitch \
        --token_joint true \
        --stage 12 \
        --stop_stage 12 \
        --datadir data \
        --dumpdir dump \
        --st_exp ${st_exp} \
        --src_lang ${tgt_lang} \
        --src_case tc \
        --tgt_lang ${tgt_lang} \
        --tgt_token_type bpe \
        --src_token_type bpe \
        --src_bpedir ${bpedir} \
        --tgt_bpedir ${bpedir} \
        --src_nbpe ${nbpe} \
        --tgt_nbpe ${nbpe} \
        --tgt_case tc \
        --token_listdir ${bpedir}/../ \
        --st_config ${st_config} \
        --inference_config conf/decode_st.yaml \
        --train_set ${train_set} \
        --valid_set ${dev_set} \
        --test_sets "${test_set}" \
        --inference_st_model valid.acc.ave.pth \
        --multilingual_mode true \
        --input_token_list_ftype token_flist \
        --lid ${tgt_lang} \
        --inference_tag "decode_st_ctc_${d}_${nbest}best" \
        --inference_args "--ctc_weight ${d} --nbest ${nbest}"
done
