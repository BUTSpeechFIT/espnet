#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).

langs=en,pt  # List of languages separated by comma ,
mlm_steps="en,pt"
text_case=tc  # caseing tc: true case, lc.rm: lowercase with punc removed

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbol

token_type="bpe"
bpemode="unigram"
nbpe=8000
token_joint=true    # whether to use a single bpe system for all the languages
bpe_char_cover=1.0  # character coverage when modeling BPE for languages
bpe_nlsyms="<mask>"  # non-linguistic symbols list, separated by a comma, for BPE
bpe_input_sentence_size=1000000  # 1M
bpe_dir=

# Language model related
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the directory path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034

ignore_init_mismatch=false      # Ignore initial mismatch
num_splits_mt=1            # Number of splitting for lm corpus.

# Upload model related
hf_repo=

batch_size=1
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
# lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --langs <list of languages> --lm_train_text <lm_train_text> --lm_dev_text  <lm_dev_text>"

Options:
    # General configuration
    --langs          # list of languages separated by comma
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").

    # Input language related
    --langs          # languages separated by comma
    --text_case      # text case tc or lc.rm

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --token_joint=true        # Whether to use a single bpe system for all languages.
                              # if set as true, will use tgt_* for processing (default="${token_joint}").
    --token_type=bpe          # Tokenization type (char or bpe) for source languages. (default="${token_type}").
    --nbpe=8000               # The number of BPE vocabulary for source languages. (default="${nbpe}").
    --bpemode=unigram         # Mode of BPE for source language (unigram or bpe). (default="${bpemode}").
    --bpe_input_sentence_size=100000000 # Size of input sentence for BPE for each  language. (default="${bpe_input_sentence_size}").
    --bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE of source language. (default="${bpe_nlsyms}").
    --bpe_char_cover=1.0  # Character coverage when modeling BPE for language. (default="${bpe_char_cover}").
    --bpe_dir          # Path to save BPE model

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --download_model             # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --lm_train_text=  # Text file path of language model training set
    --lm_dev_text=    # Text file path of language model development set.
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${lm_train_text}" ] && { log "${help_message}"; log "Error: --lm_train_text is required"; exit 2; };
[ -z "${lm_dev_text}" ] && { log "${help_message}"; log "Error: --lm_dev_text is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi


if "${token_joint}"; then
    # if token_joint, the bpe training will use both
    mkdir -pv ${bpe_dir}
    bpe_model="${bpe_dir}/bpe.model"
    token_list="${bpe_dir}/tokens.txt"
    bpeprefix=${bpe_dir}/${token_type}
else
    echo "Not implemented yet"
    exit;
fi

# NOTE: keep for future development.
# shellcheck disable=SC2034
#tgt_wordtoken_list="${token_listdir}"/word/tgt_tokens.txt
#if "${token_joint}"; then
#    src_wordtoken_list="${tgt_wordtoken_list}"
#else
#    src_wordtoken_list="${token_listdir}"/word/src_tokens.txt
#fi

# Set token types for src and tgt langs
#if [ "${token_type}" = bpe ]; then
#    token_list="${bpe_token_list}"
#elif [ "${src_token_type}" = char ]; then
#    src_token_list="${src_chartoken_list}"
#    src_bpemodel=none
#elif [ "${src_token_type}" = word ]; then
#    src_token_list="${src_wordtoken_list}"
#    src_bpemodel=none
#else
#    log "Error: not supported --token_type '${token_type}'"
#    exit 2
#fi
# if [ "${tgt_token_type}" = bpe ]; then
#     tgt_token_list="${tgt_bpetoken_list}"
# elif [ "${tgt_token_type}" = char ]; then
#     tgt_token_list="${tgt_chartoken_list}"
#     tgt_bpemodel=none
# elif [ "${tgt_token_type}" = word ]; then
#     tgt_token_list="${tgt_wordtoken_list}"
#     tgt_bpemodel=none
# else
#     log "Error: not supported --tgt_token_type '${tgt_token_type}'"
#     exit 2
# fi
# if ${use_word_lm}; then
#     log "Error: Word LM is not supported yet"
#     exit 2

#     token_list="${tgt_wordtoken_list}"
#     token_type=word
# else
#     token_list="${tgt_token_list}"
#     token_type="${tgt_token_type}"
# fi


if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    # if [ "${lang}" != noinfo ]; then
    #    lm_tag+="_${lang}_${token_type}"
    #else
        lm_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi


if [ -z "${lm_stats_dir}" ]; then
    lm_stats_dir="${expdir}/lm_stats_${token_type}"
    if [ "${token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}_${token_type}_${nbpe}"
fi


# # ========================== Main stages start from here. ==========================

# if ! "${skip_data_prep}"; then
#     if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#         log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
#         # [Task dependent] Need to create data.sh for new corpus
#         local/data.sh ${local_data_opts}

#     fi

#     if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#         if [ "${feats_type}" = raw ]; then
#             log "Stage 2: data/ -> ${data_feats}"

#             for dset in "${train_set}" "${valid_set}" ${test_sets}; do
#                 if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
#                     _suf="/org"
#                 else
#                     _suf=""
#                 fi
#                 mkdir -p "${data_feats}${_suf}/${dset}"

#                 for extra_file in ${utt_extra_files}; do
#                     # with regex to suuport multi-references
#                     for single_file in $(ls data/"${dset}"/${extra_file}*); do
#                         cp ${single_file} "${data_feats}${_suf}/${dset}"
#                     done
#                 done
#                 echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
#             done
#         else
#             log "Error: not supported: --feats_type ${feats_type}"
#             exit 2
#         fi
#     fi


#     if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#         log "Stage 3: Data filtering: ${data_feats}/org -> ${data_feats}"

#         # NOTE(kamo): Not applying to test_sets to keep original data
#         for dset in "${train_set}" "${valid_set}"; do
#             # Copy data dir
#             mkdir -p "${data_feats}/${dset}"
#             cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

#             for utt_extra_file in ${utt_extra_files}; do
#                 cp "${data_feats}/org/${dset}/${utt_extra_file}" "${data_feats}/${dset}"
#             done
#             # TODO: Maybe Remove empty text
#             # TODO: Add other data cleaning -- currently being done as part of data.sh
#         done

#         # shellcheck disable=SC2002
#         cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
#     fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

#         if "${token_joint}"; then
#             log "Merge src and target data if joint BPE"

#             cat $tgt_bpe_train_text > ${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}
#             [ ! -z "${src_bpe_train_text}" ] && cat ${src_bpe_train_text} >> ${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}
#             # Set the new text as the target text
#             tgt_bpe_train_text="${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}"
#         fi

    if [ "${token_type}" = bpe ]; then
        log "Stage 4a: Generate token_list from ${lm_train_text} using BPE"

        mkdir -p "${bpe_dir}"

        if [ -n "${bpe_nlsyms}" ]; then
            _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
        else
            _opts_spm=""
        fi

        spm_train \
            --input="${lm_train_text}" \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpeprefix}" \
            --character_coverage=${bpe_char_cover} \
            --input_sentence_size="${bpe_input_sentence_size}" \
            ${_opts_spm}

        {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
        } > "${token_list}"

    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi

fi

# ========================== Data preparation is done here. ==========================

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    if [ "${plm_type}" == "xlm" ]; then
        log "Stage 5: LM collect stats: train_set=${lm_train_text}, dev_set=${lm_dev_text}"

        _opts=
        if [ -n "${lm_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
            _opts+="--config ${lm_config} "
        fi

        # 1. Split the key file
        _logdir="${lm_stats_dir}/logdir"
        mkdir -p "${_logdir}"
        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${lm_train_text} wc -l)" "$(<${lm_dev_text} wc -l)")

        key_file="${lm_train_text}"
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${lm_dev_text}"
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/dev.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"

        # 3. Submit jobs
        log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.
        # shellcheck disable=SC2086
        ${local_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                     ${python} -m espnet2.bin.xlm_train \
                     --collect_stats true \
                     --use_preprocessor true \
                     --bpemodel "${bpe_model}" \
                     --token_type "${token_type}"\
                     --token_list "${token_list}" \
                     --non_linguistic_symbols "${nlsyms_txt}" \
                     --cleaner "${cleaner}" \
                     --g2p "${g2p}" \
                     --train_data_path_and_name_and_type "${lm_train_text},text,text" \
                     --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                     --train_shape_file "${_logdir}/train.JOB.scp" \
                     --valid_shape_file "${_logdir}/dev.JOB.scp" \
                     --output_dir "${_logdir}/stats.JOB" \
                     --langs ${langs} \
                     --mlm_steps ${mlm_steps} \
                     ${_opts} ${lm_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${lm_stats_dir}/train/text_shape" \
         awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
         >"${lm_stats_dir}/train/text_shape.${token_type}"

        <"${lm_stats_dir}/valid/text_shape" \
         awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
         >"${lm_stats_dir}/valid/text_shape.${token_type}"

    elif [ "${plm_type}" == "hlm" ]; then

        echo "run data prep and stats from mt"
        exit;

    fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: MLM Training: train_set=${lm_train_text}, dev_set=${lm_dev_text}"

    _opts=
    if [ -n "${lm_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
        _opts+="--config ${lm_config} "
    fi

    if [ "${num_splits_lm}" -gt 1 ]; then
        # If you met a memory error when parsing text files, this option may help you.
        # The corpus is split into subsets and each subset is used for training one by one in order,
        # so the memory footprint can be limited to the memory required for each dataset.

        _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            ${python} -m espnet2.bin.split_scps \
                      --scps "${data_feats}/lm_train.txt" "${lm_stats_dir}/train/text_shape.${token_type}" \
                      --num_splits "${num_splits_lm}" \
                      --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi

        _opts+="--train_data_path_and_name_and_type ${lm_train_text},text,text "
        _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
        _opts+="--multiple_iterator true "

    else
        _opts+="--train_data_path_and_name_and_type ${lm_train_text},text,text "
        _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${token_type} "
    fi

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

    log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 7 using this script"
    mkdir -p "${lm_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

    log "MLM training started... log: '${lm_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${lm_exp})"
    else
        jobname="${lm_exp}/train.log"
    fi

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
              --cmd "${cuda_cmd} --name ${jobname}" \
              --log "${lm_exp}"/train.log \
              --ngpu "${ngpu}" \
              --num_nodes "${num_nodes}" \
              --init_file_prefix "${lm_exp}"/.dist_init_ \
              --multiprocessing_distributed true -- \
              ${python} -m espnet2.bin.xlm_train \
              --ngpu "${ngpu}" \
              --use_preprocessor true \
              --bpemodel "${bpe_model}" \
              --token_type "${token_type}"\
              --token_list "${token_list}" \
              --non_linguistic_symbols "${nlsyms_txt}" \
              --cleaner "${cleaner}" \
              --g2p "${g2p}" \
              --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
              --valid_shape_file "${lm_stats_dir}/valid/text_shape.${token_type}" \
              --fold_length "${lm_fold_length}" \
              --resume true \
              --output_dir "${lm_exp}" \
              --langs ${langs} \
              --mlm_steps ${mlm_steps} \
              ${_opts} ${lm_args}

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Calc perplexity: ${lm_test_text}"
    _opts=
    # TODO(kamo): Parallelize?
    log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
                ${python} -m espnet2.bin.lm_calc_perplexity \
                --ngpu "${ngpu}" \
                --data_path_and_name_and_type "${lm_test_text},text,text" \
                --train_config "${lm_exp}"/config.yaml \
                --model_file "${lm_exp}/${inference_lm}" \
                --output_dir "${lm_exp}/perplexity_test" \
                --langs ${langs} \
                --mlm_steps ${mlm_steps} \
                ${_opts}
    log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

fi


if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    log "Stage 14: Upload model to Zenodo: ${packed_model}"

    # To upload your model, you need to do:
        #   1. Sign up to Zenodo: https://zenodo.org/
        #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
        #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="
git checkout $(git show -s --format=%H)"

    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/st1/ -> foo/st1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/st1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # Generate description file
    cat << EOF > "${mt_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${mt_exp}"/RESULTS.md)</code></pre></li>
<li><strong>MT config</strong><pre><code>$(cat "${mt_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

    # NOTE(kamo): The model file is uploaded here, but not published yet.
    #   Please confirm your record at Zenodo and publish it by yourself.

    # shellcheck disable=SC2086
    # espnet_model_zoo_upload \
    #     --file "${packed_model}" \
    #     --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
    #     --description_file "${mt_exp}"/description \
    #     --creator_name "${_creator_name}" \
    #     --license "CC-BY-4.0" \
    #     --use_sandbox false \
    #     --publish false
else
    log "Skip the uploading stages"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace" && \
            exit 1
        log "Stage 15: Upload model to HuggingFace: ${hf_repo}"

        gitlfs=$(git lfs --version 2> /dev/null || true)
        [ -z "${gitlfs}" ] && \
            log "ERROR: You need to install git-lfs first" && \
            exit 1

        dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
        [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="git checkout $(git show -s --format=%H)"
        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=machine-translation
        # shellcheck disable=SC2034
        espnet_task=MT
        # shellcheck disable=SC2034
        task_exp=${mt_exp}
        eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

        this_folder=${PWD}
        cd ${dir_repo}
        if [ -n "$(git status --porcelain)" ]; then
            git add .
            git commit -m "Update model"
        fi
        git push
        cd ${this_folder}
    fi
else
    log "Skip the uploading to HuggingFace stage"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
