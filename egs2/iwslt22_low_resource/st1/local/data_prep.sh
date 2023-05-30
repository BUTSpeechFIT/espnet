#!/usr/bin/env bash

export LC_ALL=C

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <orig_iwslt_taq_fr_dir>"
    echo "e.g.: $0 IWSLT2022_Tamasheq_data/taq_fra_clean/"
    exit 1;
fi
dataset_dir=$1 # path to dataset raw data

train_set=train
test_set=test
valid_set=valid

datadir=data/  # path where prepared data for experiment will be




python3 data_prep.py --train_set ${train_set}   \
                     --test_set ${test_set}     \
                     --valid_set ${valid_set}   \
                     --dset_path ${dataset_dir} \
                     --datadir ${datadir}
