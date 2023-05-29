#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 04 Apr 2023
# Last modified : 04 Apr 2023

"""
Manually download the data from: https://ai4bharat.org/shrutilipi
Then run this script that will prepare wav.scp, text from Shrutilipi data.
"""

import os
import sys
import argparse


def load_transcription(fname, thresh):
    # could be done with csv, but might lead to some exceptions, didn't check
    texts = []
    selected_ixs = []
    line_num = 0
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line_num += 1
            score, rest = line.strip().split(",", maxsplit=1)
            if int(score) < thresh:
                continue

            selected_ixs.append(line_num)
            parts = rest.strip().rsplit(",", maxsplit=8)
            text = parts[0].strip()

            try:
                _ = int(parts[1].strip())
            except ValueError:
                print("\nError. expected 7 integers in the end", line)
                print(fname)
                sys.exit()

            texts.append(text)

    return texts, selected_ixs


def main():
    """main method"""

    args = parse_arguments()

    os.makedirs(args.out_data_dir, exist_ok=True)

    in_dir = os.path.realpath(args.in_dir)

    out_dir = os.path.realpath(args.out_data_dir)

    lang = in_dir.split("/")[-1]

    print("Lang:", lang)

    all_text = []
    wav_scps = []
    utt_ids = []

    # for each sub dir
    # load transcription.txt with score, load sent_idx.wav
    dirs = os.listdir(in_dir)
    print("Found", len(dirs), "sub dirs")
    print("  Sample:", dirs[0])

    for k, subd in enumerate(dirs):
        subdir = os.path.join(in_dir, subd)
        trans_file = os.path.join(subdir, "transcriptions.txt")
        if os.path.exists(trans_file):
            sel_text, sel_ixs = load_transcription(trans_file, args.thresh)

            for i, ix in enumerate(sel_ixs):
                wav_fname = os.path.join(subdir, f"sent_{ix}.wav")
                if os.path.exists(wav_fname):
                    uttid = f"{lang}_{subd}_sent_{ix}"
                    ffmpeg_cmd = (
                        f"ffmpeg -i {wav_fname} -f wav -ar 16000 -ab 16 -ac 1 - | "
                    )
                    line = f"{uttid} {ffmpeg_cmd}"

                    utt_ids.append(uttid)
                    wav_scps.append(line)

                    all_text.append(f"{uttid} {sel_text[i]}")

        else:
            continue

        assert len(wav_scps) == len(
            all_text
        ), "wav_scps ({:d}) != all_text ({:d}) so far".format(
            len(wav_scps), len(all_text)
        )

        print(
            "\rProcessed subdir {:6d} / {:6d} | Found {:9d} utterances".format(
                k + 1, len(dirs), len(wav_scps)
            ),
            end=" ",
        )

    print("\nFound", len(wav_scps), "utterances in total.")

    # write to disk
    with open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8") as fpw:
        fpw.write("\n".join(wav_scps) + "\n")

    with open(os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8") as fpw:
        for uttid in utt_ids:
            fpw.write(f"{uttid} {uttid}\n")

    with open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as fpw:
        fpw.write("\n".join(all_text) + "\n")

    print("Run utils/fix_data_dir.sh " + out_dir)


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        help="path to shrutilipi dir (eg: shrutilipi/newsonair_v5/hindi/)",
    )
    parser.add_argument(
        "--out_data_dir",
        required=True,
        help="out data dir where wav.scp, utt2spk, text will be saved",
    )
    parser.add_argument(
        "--thresh",
        default=85,
        choices=range(80, 101),
        type=int,
        help="threshold on alignment score that will be used to select the data",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
