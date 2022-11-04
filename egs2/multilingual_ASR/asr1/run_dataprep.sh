#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# General variables
dataprep_start_stage=0
dataprep_stop_stage=1

asr_config=conf/train_asr_transformer.yaml
inference_config=conf/decode.yaml

feats_type=extracted

nlsyms=data/nlsyms
bpe_nlsyms="[hes]"

use_lm=false

datadir=data
dumpdir=dump
expdir=exp
asr_tag="dataprep"

token_type=bpe
nbpe=10000
hours=50
langs="it de es pl nl fr"
utt2cat=true
new_dset_appendix=


# Preparing speech data, generating filterbanks in data/, dump/ and combining datasets
# into one train, dev, test folders with mulitple language concatenated.
if [ ${dataprep_start_stage} -le 0 ] && [ ${dataprep_stop_stage} -ge 0 ]; then

    stage=1
    stop_stage=4

    train_set="train"
    valid_set="dev"
    test_sets="test"

    echo "Stage 1 of custom data preparation.
        Running stages ${stage} to ${stop_stage} in main asr script."

    ./ml_asr_dataprep.sh                            \
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
        --bpe_train_text "data/${train_set}/text"   \
        --expdir "${expdir}"                        \
        --asr_tag "multiling_test"                  \
        --dumpdir "${dumpdir}"                      \
        --gpu_inference true                        \
        --inference_nj 1                            \
        --nj 8                                      \
        --langs "${langs}"                          \
        --create_dataset true                       \
        --build_dset_names false                     \
        "$@"
    #   --nlsyms_txt ${nlsyms}                      \
    #   --bpe_nlsyms ${bpe_nlsyms}                  \
fi

# Creating tokenizers and token_list folder for each language.
# Generating files containing paths to tokenizer model,
# vocabulary and text data based on language ID.
if [ ${dataprep_start_stage} -le 1 ] && [ ${dataprep_stop_stage} -ge 1 ]; then

    echo "Stage 2 of custom data preparation.
        Running stage 5 for ${langs} languages in main asr script.
        Creating tokenizers and path files for model, vocab and text."

    train_set="train"
    valid_set="dev"
    test_sets="test"

    for lang in ${langs}; do
        lang_dependent_tokenlist_path=${datadir}/${lang}_token_list/${token_type}_unigram${nbpe}
        if [ -d ${lang_dependent_tokenlist_path} ]; then
            echo "Folder ${token_type}_unigram${nbpe} for language "${lang}" exists, skipping."
            continue
        fi
        ./ml_asr_dataprep.sh                            \
            --stage 5                                   \
            --stop_stage 5                              \
            --feats_type ${feats_type}                  \
            --token_type ${token_type}                  \
            --nbpe ${nbpe}                              \
            --use_lm ${use_lm}                          \
            --asr_config "${asr_config}"                \
            --inference_config "${inference_config}"    \
            --train_set "${train_set}"                  \
            --valid_set "${valid_set}"                  \
            --test_sets "${test_sets}"                  \
            --bpe_train_text "data/${train_set}/text"   \
            --lm_train_text "data/${train_set}/text"    \
            --expdir "${expdir}"                        \
            --asr_tag ${asr_tag}                        \
            --dumpdir "${dumpdir}"                      \
            --gpu_inference true                        \
            --inference_nj 1                            \
            --lang ${lang}                              \
            --langs ${lang}                             \
            --create_dataset false                      \
            "$@"
    done
   
    # Token list folder name
    token_list_id_folder=${datadir}/token_list_multilingual/${token_type}_unigram${nbpe}

    # Recreate token_list folder with given languages 
    if [ -d ${token_list_id_folder} ]; then
        rm -r ${token_list_id_folder} 
    fi
    
    mkdir -pv  ${token_list_id_folder}
    lang_id=0 
    for lang in ${langs} 
    do
        lang_dependent_tokenlist_path=$PWD/${datadir}/${lang}_token_list/${token_type}_unigram${nbpe}
        echo -e "${lang_id} ${lang_dependent_tokenlist_path}/bpe.model" >> ${token_list_id_folder}/bpe.model
        echo -e "${lang_id} ${lang_dependent_tokenlist_path}/bpe.vocab" >> ${token_list_id_folder}/bpe.vocab
        echo -e "${lang_id} ${lang_dependent_tokenlist_path}/tokens.txt" >> ${token_list_id_folder}/tokens.txt
        echo -e "${lang_id} ${lang_dependent_tokenlist_path}/train.txt" >> ${token_list_id_folder}/train.txt
        echo -e "${lang_id} ${lang}" >> ${token_list_id_folder}/id2lang.txt
        let lang_id=lang_id+1
    done

fi


if [ -d ${new_dset_appendix} ]; then
    mul_dataset=$(echo ${langs} | tr " " "\n" | sort -g | tr "\n" "_")
    new_train_set_appendix=_${mul_dataset}${hours}
    new_valid_set_appendix=_${mul_dataset::-1}  # remove last '_'
else
    new_train_set_appendix=_${new_dset_appendix}${hours}
    new_valid_set_appendix=_${new_dset_appendix}
fi

# create lid.scp and utt2category symlink to it, to create batches of same language during training
if [ ${utt2cat} = true ]; then
    dump_train_dir=${dumpdir}/${feats_type}/${train_set}${new_train_set_appendix}
    dump_val_dir=${dumpdir}/${feats_type}/${valid_set}${new_valid_set_appendix}

    python3 lid_scp.py --bpedir ${token_list_id_folder} \
                       --dump_dirs ${dump_train_dir} ${dump_val_dir} 

    if [ -f ${dump_train_dir}/utt2category ]; then
        rm ${dump_train_dir}/utt2category
    fi
    if [ -f ${dump_val_dir}/utt2category ]; then
        rm ${dump_val_dir}/utt2category
    fi

    current_path=$PWD
    cd ${current_path}/${dump_train_dir}
    ln -s lid.scp utt2category
    cd ${current_path}/${dump_val_dir}
    ln -s lid.scp utt2category
fi


