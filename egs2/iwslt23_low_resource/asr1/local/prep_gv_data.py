#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 24 Mar 2023
# Last modified : 24 Mar 2023

"""
Prepare Gramvaani data for Hindi
"""

import os
import sys
import argparse
from prep_mucs_data import load_transcription


def main():
    """ main method """

    args = parse_arguments()

    os.makedirs(args.out_data_dir, exist_ok=True)

    lang = "hi"
    gv_dir = os.path.realpath(args.gv_dir)

    for split, out_map in zip(["Train_100h", "Dev_5h", "Eval_3h"], ["train", "dev", "test"]):

        print(split, "->", out_map)

        out_dir = os.path.join(args.out_data_dir, f"{out_map}_gv_{lang}")
        os.makedirs(out_dir, exist_ok=True)

        fname = os.path.join(gv_dir, f"GV_{split}/text")
        keys, data = load_transcription(fname, lang)

        wav_scps = []
        utt_ids = []
        for key, uttid in keys.items():
            audio_fname = os.path.join(gv_dir, f"GV_{split}/Audio/{key}.mp3")
            if os.path.exists(audio_fname):
                ffmpeg_cmd = f"ffmpeg -i {audio_fname} -f wav -ar 16000 -ab 16 -ac 1 - | "
                line = f"{uttid} {ffmpeg_cmd}"
                wav_scps.append(line)
                utt_ids.append(uttid)
            else:
                print(audio_fname, 'not found.')
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
    parser.add_argument("--gv_dir", required=True, help="path to Interspeech 2022 Gramvaani Hindi challenge data dir (eg: GramVaani/) ")
    parser.add_argument("--out_data_dir", required=True, help="path to output data dir where train/dev/test sub-dirs with necessary files will be created (eg: data/).")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
