
# Recipe for training bi/multilingual Hindi-Marathi ASR system

1. Data preparation - combining several Hindi, Marathi datasets

    - See `local/data.sh` - better to run steps one after the other

2. Use `run.sh` to do feature extraction, remove long-short utts, learn BPE
   and train monolingual ASR models - just like a standard ESPnet2 recipe

    - feat extract, copy to dump, remove long short

    ```bash
    ./run.sh hi conf/train_asr.yaml 8000 token_listdir/hi/ \
      exp/transformer/hi_8000bpe_12L_6L_256d_0.3ctc_0.1d_0.0005lr_100e/ 2 4
    ```

    - learn BPE

    ```bash
    ./run.sh hi conf/train_asr.yaml 8000 token_listdir/hi/ \
      exp/transformer/hi_8000bpe_12L_6L_256d_0.3ctc_0.1d_0.0005lr_100e/ 5 5
    ```

    - collect stats

    ```bash
    ./run.sh hi conf/train_asr.yaml 8000 token_listdir/hi/ \
      exp/transformer/hi_8000bpe_12L_6L_256d_0.3ctc_0.1d_0.0005lr_100e/ 10 10
    ```

    - train mono ASR

    ```bash
    ./run.sh hi conf/train_asr.yaml 8000 token_listdir/hi/ \
      exp/transformer/hi_8000bpe_12L_6L_256d_0.3ctc_0.1d_0.0005lr_100e/ 11 11
    ```

    - decode and score mono ASR

    ```bash
    ./run.sh hi conf/train_asr.yaml 8000 token_listdir/hi/ \
      exp/transformer/hi_8000bpe_12L_6L_256d_0.3ctc_0.1d_0.0005lr_100e/ 12 13
    ```

    - repeat the above for Marathi (mr)

3. To train multilingual ASR models

    - Make sure the monolingual dump dirs have `lid.scp` file, if not create them.

    ```bash
    for lang in hi mr; do
      for set_name in train dev; do
        pyscripts/utils/create_lid_scp.py dump/fbank_pitch/${set_name}_${lang}/utt2dur ${lang} ;
      done
    done
    ```

    - Megre the dump dir / feats of monolingual datasets (`run.sh` for mono should be done).

    ```python3
    pyscripts/utils/merge_training_sets.py \
      --train_dirs dump/fbank_pitch/train_hi/ dump/fbank_pitch/train_mr/ \
      --out_dir dump/fbank_pitch/train_hi_mr/ \
      --utt2category hi mr
    ```

    ```python3
    pyscripts/utils/merge_training_sets.py \
      --train_dirs dump/fbank_pitch/dev_hi/ dump/fbank_pitch/dev_mr/ \
      --out_dir dump/fbank_pitch/dev_hi_mr/ \
      --utt2category hi mr
    ```

    - Create BPE flist, by combining the BPEs of monolingual parts

    ```python3
    pyscripts/utils/create_token_flist.py \
      -token_listdirs token_list/hi/bpe_unigram8000/ token_list/mr/bpe_unigram8000/ \
      -categories hi mr \
      -out_token_listdir token_list/hi_mr/bpe_unigram8000/
    ```

4. Use `run_asr_multiling.sh` with steps `collect_stats`, `train`, `decode` and `score` only.

    - collect stats

   ```bash
   run_asr_multiling.sh hi_mr 8000 conf/train_asr_512d_ctc_0.3.yaml \
     exp/hi_mr_8000nbpe_12L_512d_6L_0.1drop_0.3ctc_100e/ collect_stats
   ```

   - train bilingual/multilingual ASR

   ```bash
   run_asr_multiling.sh hi_mr 8000 conf/train_asr_512d_ctc_0.3.yaml \
     exp/hi_mr_8000nbpe_12L_512d_6L_0.1drop_0.3ctc_100e/ train
   ```

   - Monolingual decode from bi/multilingual model

   ```bash
   for lang in hi mr; do \
      run_asr_multiling.sh hi_mr 8000 conf/train_asr_512d_ctc_0.3.yaml \
        exp/hi_mr_8000nbpe_12L_512d_6L_0.1drop_0.3ctc_100e/ decode ${lang} ; \
   done
   ```

   - Score or compute WER

   ```bash
   for lang in hi mr; do \
      run_asr_multiling.sh hi_mr 8000 conf/train_asr_512d_ctc_0.3.yaml \
        exp/hi_mr_8000nbpe_12L_512d_6L_0.1drop_0.3ctc_100e/ score ${lang} \
        token_list/${lang}/bpe_unigram8000/ ;
   done
   ```

- The bi/multilingual ASR should have slightly better WER, CER than the monolingual models.

- The multilingual ASR can now be used as initialization for speech translation fine-tuning. See [../st1/README.md](../st1/README.md) for instructions.

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