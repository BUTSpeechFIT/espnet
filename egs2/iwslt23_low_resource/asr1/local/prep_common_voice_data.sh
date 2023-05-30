#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
COMMONVOICE=

if [ $# -ne 3 ]; then
    echo "Usage: $0 <Common_voice_12_sub_dir> <lang> <data_dir/>"
    echo "  lang: hi, mr"
    echo "  eg: $0 Mozilla_Common_Voice/cv-corpus-12.0-2022-12-07/ hi data_indep/"
    exit;
fi

COMMONVOICE=$1
lang=$2 # en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk
norm_type=tc
datadir=${3}
mkdir -p $3

 . utils/parse_options.sh || exit 1;

# base url for downloads.
# Deprecated url:https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/$lang.tar.gz
# data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/${lang}.tar.gz

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# mkdir -p ${COMMONVOICE}
# if [ -z "${COMMONVOICE}" ]; then
#    log "Fill the value of 'COMMONVOICE' of db.sh"
#    exit 1
#fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_"$(echo "${lang}" | tr - _)"
train_dev=dev_"$(echo "${lang}" | tr - _)"
test_set=test_"$(echo "${lang}" | tr - _)"

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage0: Manually download commonvoice data to ${COMMONVOICE}"
    log "Then start from stage 1"
    #log "The default data of this recipe is from commonvoice 5.1, for newer version, you need to register at \
    #     https://commonvoice.mozilla.org/"
    # local/download_and_untar.sh ${COMMONVOICE} ${data_url} ${lang}.tar.gz
    exit;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Preparing data for commonvoice"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "validated" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl "${COMMONVOICE}/${lang}" ${part} ${datadir}/"$(echo "${part}_${lang}" | tr - _)"
    done

    # remove test&dev data from validated sentences
    utils/copy_data_dir.sh ${datadir}/"$(echo "validated_${lang}" | tr - _)" ${datadir}/${train_set}
    utils/filter_scp.pl --exclude ${datadir}/${train_dev}/wav.scp ${datadir}/${train_set}/wav.scp > ${datadir}/${train_set}/temp_wav.scp
    utils/filter_scp.pl --exclude ${datadir}/${test_set}/wav.scp ${datadir}/${train_set}/temp_wav.scp > ${datadir}/${train_set}/wav.scp
    utils/fix_data_dir.sh ${datadir}/${train_set}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo "norm_type is: "${norm_type}

    # normalize
    for part in "train" "test" "dev"; do
      src=${datadir}/$(echo "${part}_${lang}" | tr - _)
      dst=norm_text/$(echo "${part}_${lang}" | tr - _)
      [[ -d "${dst}" ]] && rm -rf ${dst}
      mkdir -p norm_text
      cp -rf ${src} ${dst}
      cut -f 2- -d " " ${dst}/text > ${dst}/pure_text
      awk '{print $1}' ${dst}/text > ${dst}/id

      # normalize punctuation
      normalize-punctuation.perl -l ${lang} < ${dst}/pure_text > ${dst}/${lang}.norm

      # lowercasing
      lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
      cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

      # remove punctuation
      remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

      # tokenization
      cat ${dst}/${lang}.norm | sacremoses -l ${lang} tokenize > ${dst}/${lang}.norm.tc.tok
      cat ${dst}/${lang}.norm.lc | sacremoses -l ${lang} tokenize > ${dst}/${lang}.norm.lc.tok
      cat ${dst}/${lang}.norm.lc.rm | sacremoses -l ${lang} tokenize > ${dst}/${lang}.norm.lc.rm.tok

      paste -d " " ${dst}/id ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
      paste -d " " ${dst}/id ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}
      paste -d " " ${dst}/id ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}

      cp ${dst}/text.lc.${lang} $src/text.lc
      cp ${dst}/text.lc.rm.${lang} $src/text.lc.rm
      cp ${dst}/text.tc.${lang} $src/text.tc

      cp -v ${dst}/text.${norm_type}.${lang} $src/text

    done

fi



log "Successfully finished. [elapsed=${SECONDS}s]"
