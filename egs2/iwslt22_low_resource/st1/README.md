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

```bibtex
@inproceedings{kesiraju23_interspeech,
  author    = {Santosh Kesiraju and Marek Sarvaš and Tomáš Pavlíček and
               Cécile Macaire and Alejandro Ciuba},
  title     = {{Strategies for Improving Low Resource Speech to Text
               Translation Relying on Pre-trained ASR Models}},
  year      = 2023,
  booktitle = {Proc. INTERSPEECH 2023},
  pages     = {2148--2152},
  doi       = {10.21437/Interspeech.2023-2506}
}

@inproceedings{kesiraju-etal-2023-systems,
    title     = "{BUT} Systems for {IWSLT} 2023 {M}arathi - {H}indi Low Resource
                Speech Translation Task",
    author    = "Kesiraju, Santosh  and Bene{\v{s}}, Karel  and
                 Tikhonov, Maksim and {\v{C}}ernock{\'y}, Jan",
    booktitle = "Proceedings of the 20th International Conference on Spoken
                 Language Translation (IWSLT 2023)",
    month     = jul,
    year      = "2023",
    address   = "Toronto, Canada (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2023.iwslt-1.19",
    doi       = "10.18653/v1/2023.iwslt-1.19",
    pages     = "227--234",
    abstract  = "This paper describes the systems submitted for Marathi to Hindi low-resource speech translation task. Our primary submission is based on an end-to-end direct speech translation system, whereas the contrastive one is a cascaded system. The backbone of both the systems is a Hindi-Marathi bilingual ASR system trained on 2790 hours of imperfect transcribed speech. The end-to-end speech translation system was directly initialized from the ASR, and then fine-tuned for direct speech translation with an auxiliary CTC loss for translation. The MT model for the cascaded system is initialized from a cross-lingual language model, which was then fine-tuned using 1.6 M parallel sentences. All our systems were trained from scratch on publicly available datasets. In the end, we use a language model to re-score the n-best hypotheses. Our primary submission achieved 30.5 and 39.6 BLEU whereas the contrastive system obtained 21.7 and 28.6 BLEU on official dev and test sets respectively. The paper also presents the analysis on several experiments that were conducted and outlines the strategies for improving speech translation in low-resource scenarios.",
}
```