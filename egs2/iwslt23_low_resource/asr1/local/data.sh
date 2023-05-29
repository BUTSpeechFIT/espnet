#!/bin/bash

# This script will attempt to prepare the data from all the datasets
# For simplicity it assumes the data dir are hardcoded.
# However, it should be straightforward to point those to appropriate locations


COMMONVOICE_DIR="Mozilla_Common_Voice/cv-corpus-12.0-2022-12-07"

# Get it from https://ai4bharat.org/shrutilipi
SHRUTILIPI_DIR="Shrutilipi/newsonair_v5/"

# data from Interspeech 2022
GRAMVAANI_DIR="GramVaani/"

# MUCS data from Interpseech 2021 - should contain Hindi, Marathi sub-dirs
# which inturn should contain train, test sub-sub-dirs
# The MUCS isn't a great dataset since it has very little lexical variability.
MUCS_DIR="IS21_subtask_1_data"

# This contains data from openslr tts, and crowd source speech data
# expect sub-dirs College-Students, Rural-Low-Income-Workers, Urban-Low-Income-Workers, mr_in_female
MARATHI_MISC_DIR="Marathi/"



# First we prepare for individual datasets and then finally combine all of them
bash local/prep_common_voice_data.sh ${COMMONVOICE_DIR} hi data_indiv
bash local/prep_common_voice_data.sh ${COMMONVOICE_DIR} mr data_indiv

# Prepare IS'21 MUCS data
python3 local/prep_mucs_data.py --lang hi --mucs_dir ${MUCS_DIR}/Hindi/ --out_data_dir data_indiv/
python3 local/prep_mucs_data.py --lang mr --mucs_dir ${MUCS_DIR}/Marathi/ --out_data_dir data_indiv/

# Next prepare GramVaani hi data
python3 local/prep_gv_data.py --gv_dir ${GRAMVAANI_DIR} --out_data_dir data_indiv/

# Next prepare Marathi Misc data (from tts and crowd-source)
python3 local/prep_mr_data.py --in_dir ${MARATHI_MISC_DIR} --out_data_dir data_indiv/

# Finally, prepare Shrutilipi data
python local/prep_shrutilipi_data.py --in_dir ${SHRUTILIPI_DIR}/hindi/ \
    --out_data_dir data_indiv/train_shrutilipi_hi

python local/prep_shrutilipi_data.py --in_dir ${SHRUTILIPI_DIR}/marathi/ \
    --out_data_dir data_indiv/train_shrutilipi_mr


# Merge the individual datasets and save them data/ directory

# Hindi - train
pyscripts/utils/merge_training_sets.py \
    --train_dirs data_indiv/train_hi/ data_indiv/train_mucs_hi/ \
        data_indiv/train_shrutilipi_hi/ data_indiv/train_gv_hi/ \
    --out_dir data/train_hi/ \
    --ignore_missing

utils/fix_data_dir.sh data/train_hi

# Hindi - dev
pyscripts/utils/merge_training_sets.py \
    --train_dirs data_indiv/dev_hi/ data_indiv/dev_mucs_hi/ data_indiv/dev_gv_hi/ \
    --out_dir data/dev_hi/ \
    --ignore_missing

utils/fix_data_dir.sh data/dev_hi

# Hindi - test
pyscripts/utils/merge_training_sets.py \
    --train_dirs data_indiv/test_hi/ data_indiv/test_gv_hi/ \
    --out_dir data/test_hi/ \
    --ignore_missing

utils/fix_data_dir.sh data/test_hi

# Marathi - train
pyscripts/utils/merge_training_sets.py \
    --train_dirs data_indiv/train_mr/ data_indiv/train_mucs_mr/ \
        data_indiv/train_shrutilipi_mr/  data_indiv/train_misc_mr/ \
    --out_dir data/train_mr/ \
    --ignore_missing

utils/fix_data_dir.sh data/train_mr

# Marathi - dev
pyscripts/utils/merge_training_sets.py \
    --train_dirs data_indiv/dev_mr/ data_indiv/dev_mucs_mr/ \
    --out_dir data_combined/dev_mr/ \
    --ignore_missing

utils/fix_data_dir.sh data/dev_mr

# Marathi - test
pyscripts/utils/merge_training_sets.py \
    --train_dirs data_indiv/test_mr/ \
    --out_dir data_combined/test_mr/ \
    --ignore_missing

utils/fix_data_dir.sh data/test_mr


# Merge Hindi and Marathi train and dev sets
# also create utt2category, lid.scp, utt2lang files
pyscripts/utils/merge_training_sets.py \
    --train_dirs data/train_hi/ data/train_mr/ \
    --out_dir data/train_hi_mr/ \
    --utt2category hi mr \
    --ignore_missing

utils/fix_data_dir.sh data/train_hi_mr

pyscripts/utils/merge_training_sets.py \
    --train_dirs data/dev_hi/ data/dev_mr/ \
    --out_dir data/dev_hi_mr/ \
    --utt2category hi mr \
    --ignore_missing

utils/fix_data_dir.sh data/dev_hi_mr
