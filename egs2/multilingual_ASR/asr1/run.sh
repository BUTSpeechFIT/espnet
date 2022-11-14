#!/usr/bin/env bash

set -e
set -u
set -o pipefail

if [ $# -ne 1 ]; then
    echo "$0 <stats (for collecting stats) or train (for model training)>"
    exit;
fi
step=$1


train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/train_asr_transformer.yaml
inference_config=conf/decode.yaml

feats_type=extracted
token_type=bpe

nlsyms=data/nlsyms
bpe_nlsyms="[hes]"

nbpe=10000

use_lm=false

stage=10
stop_stage=10

datadir=data
dumpdir=dump
expdir=exp

langs="de es fr it nl pl"
hours=50

asr_tag="50h_6langs_dict"
token_listdir="data/token_list_multilingual"

########### Create dataset paths from selected languages #############
langs_dataset=$(echo ${langs} | tr " " "\n" | sort -g | tr "\n" "_")
new_train_set_appendix=_${langs_dataset}${hours}
new_valid_set_appendix=_${langs_dataset::-1}

train_set=${train_set}${new_train_set_appendix}
valid_set=${valid_set}${new_valid_set_appendix}
lm_train_text=${datadir}/${train_set}/text
bpe_train_text=${datadir}/${train_set}/text

test_list="" 
for ll in ${langs}; do
    # create list of test sets for each language 
    lang_test="${test_sets}_${ll}"
    test_list="${test_list} ${lang_test}" 
done
test_sets=${test_list:1}  # set variable and remove first space
echo ${test_sets}
echo ${train_set}
echo ${valid_set}
#####################################################################

if [ "${step}" == "stats" ]; then

    ./asr.sh                                        \
        --stage ${stage}                            \
        --stop_stage ${stop_stage}                  \
        --feats_type ${feats_type}                  \
        --token_type ${token_type}                  \
        --nbpe ${nbpe}                              \
        --use_lm ${use_lm}                          \
        --asr_config "${asr_config}"                \
        --inference_config "${inference_config}"    \
        --train_set "${train_set}"                  \
        --valid_set "${valid_set}"                  \
        --test_sets "${test_sets}"                  \
        --bpe_train_text "${bpe_train_text}"        \
        --lm_train_text "${lm_train_text}"          \
        --expdir "${expdir}"                        \
        --token_listdir ${token_listdir}            \
        --asr_tag ${asr_tag}                        \
        --dumpdir "${dumpdir}"                      \
        --gpu_inference true                        \
        --inference_nj 1                            \
        --nj 8                                      \
        --multilingual_mode true                    
fi

if [ "${step}" == "train" ]; then

    ./asr.sh                                        \
        --stage 11                                  \
        --stop_stage 11                             \
        --feats_type ${feats_type}                  \
        --token_type ${token_type}                  \
        --nbpe ${nbpe}                              \
        --use_lm ${use_lm}                          \
        --asr_config "${asr_config}"                \
        --inference_config "${inference_config}"    \
        --train_set "${train_set}"                  \
        --valid_set "${valid_set}"                  \
        --test_sets "${test_sets}"                  \
        --bpe_train_text "${bpe_train_text}"        \
        --lm_train_text "${lm_train_text}"          \
        --expdir "${expdir}"                        \
        --token_listdir ${token_listdir}            \
        --asr_tag ${asr_tag}                        \
        --dumpdir "${dumpdir}"                      \
        --gpu_inference true                        \
        --inference_nj 1                            \
        --nj 8                                      \
        --multilingual_mode true                    \
        --copy_feats_to_dir /mnt/ssd/               
fi
