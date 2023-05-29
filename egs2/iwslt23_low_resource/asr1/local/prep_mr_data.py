#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 24 Mar 2023
# Last modified : 24 Mar 2023

"""
Prepare Marathi data from various sources. Download and extarct in same subdir
IIT-B: https://www.cse.iitb.ac.in/~pjyothi/indiccorpora/
TTS datasets from openslr: https://www.openslr.org/64/
"""

import os
import sys
import argparse
import glob
from prep_mucs_data import load_transcription


def prep_tts_data(args, out_dir, mode):

    in_dir = os.path.realpath(args.in_dir)

    tts_dir = "mr_in_female"

    keys, data = load_transcription(
        os.path.join(in_dir, "line_index.tsv"), "mr", delim="\t"
    )

    wav_scps = []
    utt_ids = []
    for key, uttid in keys.items():

        audio_fname = os.path.join(args.in_dir, f"{tts_dir}/{key}.wav")
        if os.path.exists(audio_fname):
            ffmpeg_cmd = f"ffmpeg -i {audio_fname} -f wav -ar 16000 -ab 16 -ac 1 - | "
            line = f"{uttid} {ffmpeg_cmd}"
            wav_scps.append(line)
            utt_ids.append(uttid)
        else:
            print(audio_fname, "not found.")
            sys.exit()

    print(len(data), "tts utterances appended to", out_dir)

    # write to disk
    with open(os.path.join(out_dir, "wav.scp"), mode) as fpw:
        fpw.write("\n".join(wav_scps) + "\n")

    with open(os.path.join(out_dir, "utt2spk"), mode) as fpw:
        for uttid in utt_ids:
            fpw.write(f"{uttid} {uttid}\n")

    with open(os.path.join(out_dir, "text"), mode, encoding="utf-8") as fpw:
        fpw.write("\n".join(data) + "\n")

    os.system("utils/fix_data_dir.sh " + out_dir)


def prep_iitb_data(args, out_dir, mode):

    in_dir = os.path.realpath(args.in_dir)

    # sub_dirs = ["College-Students", "Rural-Low-Income-Workers", "Urban-Low-Income-Workers"]

    wav_scps = []
    data = []
    utt_ids = []

    audio_fnames = glob.glob(in_dir + "/*/*.3gp")
    print("Found", len(audio_fnames), "3gp audio files")

    for i, audio_fname in enumerate(audio_fnames):
        base = os.path.basename(audio_fname).rsplit(".", 1)[0]
        subd = os.path.dirname(audio_fname).split("/")[-1]
        uttid = f"{subd}_{base}"

        print("\r {:7d} / {:7d} {:s}".format(i + 1, len(audio_fnames), uttid), end=" ")

        text_fname = audio_fname.replace(".3gp", ".txt")
        text = ""

        if os.path.exists(text_fname):
            with open(text_fname, "r", encoding="utf-8") as fpr:
                text = fpr.read().strip()
        else:
            print(text_fname, "not found.")
            sys.exit()

        ffmpeg_cmd = f"ffmpeg -i {audio_fname} -f wav -ar 16000 -ab 16 -ac 1 - | "
        line = f"{uttid} {ffmpeg_cmd}"
        wav_scps.append(line)

        data.append(f"{uttid} {text}")

        utt_ids.append(uttid)

    # write to disk
    print("\n", len(data), "utterances. Writing to", out_dir)

    with open(os.path.join(out_dir, "wav.scp"), mode) as fpw:
        fpw.write("\n".join(wav_scps) + "\n")

    with open(os.path.join(out_dir, "utt2spk"), mode) as fpw:
        for uttid in utt_ids:
            fpw.write(f"{uttid} {uttid}\n")

    with open(os.path.join(out_dir, "text"), mode, encoding="utf-8") as fpw:
        fpw.write("\n".join(data) + "\n")

    os.system("utils/fix_data_dir.sh " + out_dir)


def main():
    """main method"""

    args = parse_arguments()

    os.makedirs(args.out_data_dir, exist_ok=True)

    lang = "mr"
    out_dir = os.path.join(args.out_data_dir, f"train_misc_{lang}")
    os.makedirs(out_dir, exist_ok=True)

    prep_iitb_data(args, out_dir, mode="w")

    prep_tts_data(args, out_dir, mode="a")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        help="path to dir where Marathi data from openslr are extracted (the dir should contain sub folder College-Students, Rural-Low-Income-Workers, Urban-Low-Income-Workers,  mr_in_female).",
    )
    parser.add_argument(
        "--out_data_dir",
        required=True,
        help="path to output data dir where train/dev/test sub-dirs with necessary files will be created (eg: data/).",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
