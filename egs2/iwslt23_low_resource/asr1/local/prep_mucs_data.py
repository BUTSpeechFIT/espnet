#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 24 Mar 2023
# Last modified : 24 Mar 2023

"""
Prepare MUCS data for Hindi, Marathi
"""

import os
import sys
import argparse


def load_transcription(fname, lang, delim=" "):

    keys = {}
    data = []
    with open(fname, 'r', encoding='utf-8') as fpr:
        for line in fpr:
            key, text = line.strip().split(delim, maxsplit=1)
            uttid = f"{lang}_{key}"
            keys[key] = uttid
            data.append(f"{uttid} {text}")
    return keys, data


def main():
    """ main method """

    args = parse_arguments()

    os.makedirs(args.out_data_dir, exist_ok=True)

    lang = args.lang
    mucs_dir = os.path.realpath(args.mucs_dir)

    for split, out_map in zip(["train", "test"], ["train", "dev"]):

        print(split, "->", out_map)

        out_dir = os.path.join(args.out_data_dir, f"{out_map}_mucs_{lang}")
        os.makedirs(out_dir, exist_ok=True)

        fname = os.path.join(mucs_dir, f"{split}/transcription.txt")
        keys, data = load_transcription(fname, lang)

        wav_scps = []
        utt_ids = []
        for key, uttid in keys.items():
            wav_fname = os.path.join(mucs_dir, f"{split}/audio/{key}.wav")
            if os.path.exists(wav_fname):
                ffmpeg_cmd = f"ffmpeg -i {wav_fname} -f wav -ar 16000 -ab 16 -ac 1 - | "
                line = f"{uttid} {ffmpeg_cmd}"
                wav_scps.append(line)
                utt_ids.append(uttid)
            else:
                print(wav_fname, 'not found.')
                sys.exit()

        with open(os.path.join(out_dir, "wav.scp"), "w") as fpw:
            fpw.write("\n".join(wav_scps) + "\n")

        with open(os.path.join(out_dir, "utt2spk"), "w") as fpw:
            for uttid in utt_ids:
                fpw.write(f"{uttid} {uttid}\n")

        with open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as fpw:
            fpw.write("\n".join(data) + "\n")

        os.system("utils/fix_data_dir.sh " + out_dir)


def parse_arguments():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lang", required=True, choices=["mr", "hi"], help="Language code (eg: hi)")
    parser.add_argument("--mucs_dir", required=True, help="path to Interspeech 2021 MUCS challenge data dir (eg: IS21_subtask_1_data/Hindi/) ")
    parser.add_argument("--out_data_dir", required=True, help="path to output data dir where train/dev/test sub-dirs with necessary files will be created (eg: data/).")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
