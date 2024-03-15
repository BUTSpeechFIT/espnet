# Recipe for training Multilingual ASR model on common voice

- The following recipe is only for 6 languages (de, es, fr, it, pl, pt), where we take 50hr from each.
This is inline with the Interspeech'23 paper.

1. Data prepare from Mozilla Common Voice v8

    ```bash
    LANGS=("de" "es" "fr" "it" "pl" "pt")
    for lang in ${LANGS[@]}; do
        local/data.sh mozilla_common_voice-8.0-2022-01-19/ ${lang} tc
    done
    ```

2. Select only 50 hr subset from each language. The correspoding utterance IDs are in `subset_uttids` dir

    ```bash
    mkdir -p data_subset

    for lang in ${LANGS[@]}; do
        pyscripts/utils/create_train_subset.py \
          data/train_${lang}/ \
          subset_uttids/${lang}/train.utt2spk \
          data_subset/train_${lang}.tc/
    done
    ```

    - Retain the original dev and test sets

    ```bash
    for lang in ${LANGS[@]}; do
        for set_name in dev test; do
            ln -svf $(realpath data/${set_name}_${lang}) data_subset/${set_name}_${lang}.tc ;
        done
    done
    ```

3. For each language, extract features, learn BPE, optionally train monolingual ASR models and decode. This follows a standard ESPnet2 recipe. An example is below:

    ```bash
    for lang in ${LANGS[@]}; do
        ./run.sh data_subset/ \
                ${lang}.tc \
                conf/train_asr_ctc0.9.yaml \
                1000 \
                token_list/${lang}.50.tc/ \
                exp/asr_${lang}.50.tc_12L_256d_6L_0.1d_0.9ctc_200e/ \
                3 5
    done
    ```

    - Optional training of monolingual models

    ```bash
    for lang in ${LANGS[@]}; do
        ./run.sh data_subset/ \
                ${lang}.tc \
                conf/train_asr_ctc0.9.yaml \
                1000 \
                token_list/${lang}.50.tc/ \
                exp/asr_${lang}.50.tc_12L_256d_6L_0.1d_0.9ctc_200e/ \
                10 11
    done
    ```

    - Optional decode and score monolingual models

    ```bash
    for lang in ${LANGS[@]}; do
        ./run.sh data_subset/ \
                ${lang}.tc \
                conf/train_asr_ctc0.9.yaml \
                1000 \
                token_list/${lang}.50.tc/ \
                exp/asr_${lang}.50.tc_12L_256d_6L_0.1d_0.9ctc_200e/ \
                12 13
    done
    ```

4. Create multilingual training set by reusing the above monolingual features and BPE models

    - Making sure that each language dir has `lid.scp` `utt2category` and `utt2lang`
    - Additionally, you can append `utt2category` in the for-loop list at `line 57` in `utils/data/fix_data_dir.sh`


    ```bash
    for set_name in train dev; do
      for lang in ${LANGS[@]}; do
          pyscripts/utils/create_lid_scp.py \
            dump/fbank_pitch/${set_name}_${lang}.tc/utt2spk \
            ${lang}
      done
    done
    ```

    - Merge training and dev sets. The order of input dirs and `utt2category` arg should match.

    ```bash
    for set_name in train dev; do
        pyscripts/utils/merge_training_sets.py --train_dirs \
          dump/fbank_pitch/${set_name}_de.tc dump/fbank_pitch/${set_name}_es.tc \
          dump/fbank_pitch/${set_name}_fr.tc dump/fbank_pitch/${set_name}_it.tc \
          dump/fbank_pitch/${set_name}_pl.tc dump/fbank_pitch/${set_name}_pt.tc \
          --out_dir dump/fbank_pitch/${set_name}_6L.300.tc/ \
          --utt2category de es fr it pl pt ;

        utils/fix_data_dir.sh dump/fbank_pitch/${set_name}_6L.300.tc ;

    done
    ```

    - Create flist linking all the mono BPE models

    ```python
    pyscripts/utils/create_token_flist.py -token_listdirs \
      token_list/de.50.tc/bpe_unigram1000/ token_list/es.50.tc/bpe_unigram1000/ \
      token_list/fr.50.tc/bpe_unigram1000/ token_list/it.50.tc/bpe_unigram1000/ \
      token_list/pl.50.tc/bpe_unigram1000/ token_list/pt.50.tc/bpe_unigram1000/ \
      -categories de es fr it pl pt \
      -out_token_listdir token_list/6L.300.tc/bpe_unigram1000/
    ```

5. Train the multilingual model

   - collect stats

   ```bash
   ./run_asr_multiling.sh \
     6L.300.tc \
     1000 \
     conf/train_asr_ctc0.3.yaml \
     exp/6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/ \
     collect_stats
   ```

   - train

   ```bash
   ./run_asr_multiling.sh \
     6L.300.tc \
     1000 \
     conf/train_asr_ctc0.3.yaml \
     exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/ \
     train
   ```

   - monolingual decode (true case setup `tc`)

   ```bash
   for lang in ${LANGS[@]}; do
      ./run_asr_multiling.sh \
        6L.300.tc \
        1000 \
        conf/train_asr_ctc0.3.yaml \
        exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/ \
        decode \
        ${lang}.tc
   done
   ```

   - score, compute WER (true case setup `tc`)

   ```bash
   for lang in ${LANGS[@]}; do
      ./run_asr_multiling.sh \
        6L.300.tc \
        1000 \
        conf/train_asr_ctc0.3.yaml \
        exp/masr_6L.300.tc_1000nbpe_12L_256d_6L_0.1d_0.3ctc_100e/ \
        decode \
        ${lang}.tc \
        token_list/${lang}.50.tc/
   done
   ```

- The multilingual model should have slightly better WER than the monolingual counter-parts.

- The trained model can be used as initialization for speech translation fine-tuning. See [../st1/README.md](../st1/README.md).

```bibtex
@inproceedings{kesiraju23_interspeech,
  author    = {Santosh Kesiraju and Marek Sarvaš and Tomáš Pavlíček and
               Cécile Macaire and Alejandro Ciuba},
  title     = {{Strategies for Improving Low Resource Speech to Text
               Translation Relying on Pre-trained ASR Models}},
  year      = 2023,
  booktitle = {Proc. INTERSPEECH 2023},
  pages     = {2148--2152},
  address   = {Dublin, Ireland},
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