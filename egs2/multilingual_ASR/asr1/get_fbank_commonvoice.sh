#!/usr/bin/bash




#------------------------------------------------------------------------
. path.sh
. cmd.sh
#------------------------------------------------------------------------
#------------------------------------------------------------------------
nj=10

fbank_conf="conf/fbank.conf"
sets="train test dev"
for set_name in $sets; do
#for set_name in train "val" dev5 ; do
#for set_name in "train" ; do
#    data_dir=/mnt/matylda6/MLASR/pavlicek/projects/JSALT/data/how2-300h-v1/espnet/17h_80mfcc/${set_name}
#    highres_dir=/mnt/matylda6/MLASR/pavlicek/projects/JSALT/data/how2-300h-v1/espnet/17h_80mfcc/feas/${set_name}
    data_dir=/mnt/matylda4/xsarva00/NEUREM3/espnet/egs2/multilingual_ASR/asr1/data/${set_name}_$1
    highres_dir=/mnt/matylda4/xsarva00/NEUREM3/espnet/egs2/multilingual_ASR/asr1/data/feats/${set_name}_$1
    mkdir -p $highres_dir
    steps/make_fbank_pitch.sh --fbank-config "$fbank_conf" --cmd "$train_cmd" --nj $nj $data_dir $highres_dir $highres_dir
    steps/compute_cmvn_stats.sh $data_dir
#    bash kaldi_feature_extract_mfcc_delta_rewrite1.sh --nj $nj --cmd "$train_cmd" $data_dir $highres_dir
    #-----------------------------------

done
