1. Use `python local/prepare_data.py` to prepare the data for IWSLT'23 Marathi-Hindi

2. Run standard espnet2 steps 2, 3, 4 using `run_with_pt_masr_ctc_as_mt_sperturb.sh`
   - fbank+pitch features
   - copy to dump dir
   - remove long short utts

3. Do not create BPE if you want to use a pre-trained ASR model. See [../asr1/](../asr1/)

4. Run standard espnet2 steps 10, 11, 12, 13 using `run_with_pt_masr_ctc_as_mt_sperturb.sh`
   - extract stats - using BPE from pre-trained ASR
   - train ST model (`conf/train_st_ctc_as_mt_0.1_0.0001lr.yaml`)
   - standard decoding: dev, test sets (`conf/decode_st.yaml`)
   - score BLEU, CHRF2

5. For additional joint-decoding with $n$-best, see `decode_nbest.sh`
