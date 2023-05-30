# Recipe for training direct speech translation system relying on a pre-trained multilingual ASR

1. Use `python local/prepare_data.py` to prepare the data for IWSLT'23 Marathi-Hindi

2. Run standard espnet2 steps 2, 3, 4 using `run_with_pt_masr_ctc_as_mt_sperturb.sh`
   - speed perturbation
   - fbank+pitch feature extraction
   - copy to dump dir
   - remove long short utts

3. Do not create BPE if you want to use a pre-trained ASR model. See [../asr1/README.md](../asr1/README.md) on how to train a bi/multilingual ASR.

4. Run standard espnet2 steps 10, 11, 12, 13 using `run_with_pt_masr_ctc_as_mt_sperturb.sh`
   - collect stats - using BPE from pre-trained ASR

   ```bash
   pretrained_asr="../asr1/exp/hi_mr_8000nbpe_12L_512d_6L_0.1drop_0.3ctc_100e/valid.acc.ave.pth"

   ./run_with_pt_masr_ctc_as_mt_sperturb.sh 8000 \
      ../asr1/token_list/hi_mr/bpe_unigram8000/ \
      conf/train_st_512d_ctc_as_mt_0.1.yaml \
      conf/decode_st.yaml \
      exp/asr_init_hi_mr_8000bpe_12L_512d_6L_0.1d_0.1ctc_as_mt_sp/ \
      ${pretrained_asr} \
      mr_hi_8000bpe_sp \
      10 10
   ```

   - train ST model (`conf/train_st_512d_ctc_as_mt_0.1.yaml`)

   ```bash
   pretrained_asr="../asr1/exp/hi_mr_8000nbpe_12L_512d_6L_0.1drop_0.3ctc_100e/valid.acc.ave.pth"

   ./run_with_pt_masr_ctc_as_mt_sperturb.sh 8000 \
      ../asr1/token_list/hi_mr/bpe_unigram8000/ \
      conf/train_st_512d_ctc_as_mt_0.1.yaml \
      conf/decode_st.yaml \
      exp/asr_init_hi_mr_8000bpe_12L_512d_6L_0.1d_0.1ctc_as_mt_sp/ \
      ${pretrained_asr} \
      mr_hi_8000bpe_sp \
      11 11
   ```

   - standard decoding and scoring: dev, test sets (`conf/decode_st.yaml`)

    ```bash
   pretrained_asr="../asr1/exp/hi_mr_8000nbpe_12L_512d_6L_0.1drop_0.3ctc_100e/valid.acc.ave.pth"

   ./run_with_pt_masr_ctc_as_mt_sperturb.sh 8000 \
      ../asr1/token_list/hi_mr/bpe_unigram8000/ \
      conf/train_st_512d_ctc_as_mt_0.1.yaml \
      conf/decode_st.yaml \
      exp/asr_init_hi_mr_8000bpe_12L_512d_6L_0.1d_0.1ctc_as_mt_sp/ \
      ${pretrained_asr} \
      mr_hi_8000bpe_sp \
      12 13
   ```

5. For additional joint-decoding with $n$-best, see `dec_nbest.sh`. This should improve the results as compared to standard attention-only decoding. Use an external LM to re-score the $n$-best hypotheses should further improve. For rescoring, see [BrnoLM](https://github.com/BUTSpeechFIT/BrnoLM).

   ```bash
   ./dec_nbest.sh exp/asr_init_hi_mr_8000bpe_12L_512d_6L_0.1d_0.1ctc_as_mt_sp/ 50
   ```
