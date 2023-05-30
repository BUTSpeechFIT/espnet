# Recipe for training direct speech translation system relying on a pre-trained multilingual ASR

1. Use `bash local/data_prep.sh IWSLT2022_Tamasheq_data/taq_fra_clean/` to prepare the data for IWSLT'22 Tamasheq-French data.

2. Run standard espnet2 steps 2, 3, 4 using `run_with_pt_masr_ctc_as_mt_sperturb.sh`
   - speed perturbation (will improve the results by atleast +1.5 BLEU as compared to the Interspeech'23 paper).
   - fbank+pitch feature extraction
   - copy to dump dir
   - remove long short utts

3. Do not create BPE if you want to use a pre-trained ASR model. See [../asr1/README.md](../asr1/README.md) on how to train a multilingual ASR.

4. Run standard espnet2 steps 10, 11, 12, 13 using `run_with_pt_masr_ctc_as_mt_sperturb.sh`
   - collect stats - using BPE from pre-trained ASR

   ```bash
   pretrained_asr="../asr1/exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/valid.acc.ave.pth"

   ./run_with_pt_masr_ctc_as_mt_sperturb.sh 1000 \
      ../asr1/token_list/6L.300.tc/bpe_unigram1000/ \
      conf/train_st_ctc_as_mt_0.1.yaml \
      conf/decode_st.yaml \
      exp/masr_init_6L.300_12L_256d_6L_0.1d_0.1ctc_as_mt_200e/ \
      ${pretrained_asr} \
      6L.300.tc_sp \
      10 10
   ```

   - train ST model (`conf/train_st_ctc_as_mt_0.1.yaml`)

   ```bash
   pretrained_asr="../asr1/exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/valid.acc.ave.pth"

   ./run_with_pt_masr_ctc_as_mt_sperturb.sh 1000 \
      ../asr1/token_list/6L.300.tc/bpe_unigram1000/ \
      conf/train_st_ctc_as_mt_0.1.yaml \
      conf/decode_st.yaml \
      exp/masr_init_6L.300_12L_256d_6L_0.1d_0.1ctc_as_mt_200e/ \
      ${pretrained_asr} \
      6L.300.tc_sp \
      11 11
   ```

   - standard decoding and scoring: dev, test sets (`conf/decode_st.yaml`)

    ```bash
    pretrained_asr="../asr1/exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/valid.acc.ave.pth"

   ./run_with_pt_masr_ctc_as_mt_sperturb.sh 1000 \
      ../asr1/token_list/6L.300.tc/bpe_unigram1000/ \
      conf/train_st_ctc_as_mt_0.1.yaml \
      conf/decode_st.yaml \
      exp/masr_init_6L.300_12L_256d_6L_0.1d_0.1ctc_as_mt_200e/ \
      ${pretrained_asr} \
      6L.300.tc_sp \
      12 13
   ```

5. For additional joint-decoding with $n$-best, see `dec_nbest.sh`. This should improve the results as compared to standard attention-only decoding. Use an external LM to re-score the $n$-best hypotheses should further improve. For rescoring, see [BrnoLM](https://github.com/BUTSpeechFIT/BrnoLM).

   ```bash
   ./dec_nbest.sh exp/masr_init_6L.300_12L_256d_6L_0.1d_0.1ctc_as_mt_200e/ 50
   ```
