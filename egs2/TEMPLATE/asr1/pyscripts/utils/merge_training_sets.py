#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 01 Dec 2022
# Last modified : 01 Dec 2022

"""
Merge different training / dev / test sets into one.
Useful in creating training data for multilingual ASR
"""

import os
from copy import deepcopy
import sys
import argparse


MERGE_NAMES = [
    "text",
    "utt2dur",
    "utt2spk",
    "utt_map",
    "wav.scp",
    "feats.scp",
    "feats_shape",
    "utt2num_frames",
]
COPY_NAMES = ["feats_dim", "feats_type", "frame_shift"]


def process_line(line, cat_name):
    """Prefix text with [cat_name]"""

    utt_id, text = line.split(" ", maxsplit=1)
    processed = f"{utt_id} [{cat_name}] {text}"
    return processed


def main():
    """main method"""

    args = parse_arguments()

    flag = False
    for name in MERGE_NAMES + COPY_NAMES:
        for train_dir in args.train_dirs:
            in_file = os.path.join(train_dir, name)
            if not os.path.exists(in_file):
                print(
                    "File not found:",
                    in_file,
                    " ... Ignoring ..." if args.ignore_missing else "",
                )
                flag = True

    if flag and not args.ignore_missing:
        print("- One or more required files were missing. Cannot proceed.")
        sys.exit()

    for name in COPY_NAMES:
        content = ""
        for train_dir in args.train_dirs:
            in_file = os.path.join(train_dir, name)
            if os.path.exists(in_file):
                with open(in_file, "r", encoding="utf-8") as fpr:
                    cur_content = fpr.read().strip()
                if not content:
                    content = deepcopy(cur_content)
                else:
                    if content != cur_content:
                        print(name, "is not same across different dirs")
                        sys.exit()

    os.makedirs(args.out_dir, exist_ok=True)
    args.out_dir = os.path.realpath(args.out_dir)

    for name in COPY_NAMES:
        in_file = os.path.join(args.train_dirs[0], name)
        if os.path.exists(in_file):
            os.system("cp -v " + in_file + " " + args.out_dir + "/")

    for name in MERGE_NAMES:
        keys = {}
        content = []
        utt2cat = []
        for i, train_dir in enumerate(args.train_dirs):
            in_file = os.path.join(train_dir, name)
            if not os.path.exists(in_file):
                continue
            with open(in_file, "r", encoding="utf-8") as fpr:
                for line in fpr:
                    line = line.strip()
                    parts = line.split(" ", maxsplit=1)
                    if parts[0] not in keys:
                        keys[parts[0]] = in_file
                        if name == "text" and args.prefix_text_with_categories:
                            line = process_line(line, args.utt2category[i])
                        content.append(line)
                        if args.utt2category:
                            utt2cat.append(f"{parts[0]} {args.utt2category[i]}")
                    else:
                        if args.duplicates == "error":
                            print(
                                "- Error: Duplicate key:",
                                parts[0],
                                keys[parts[0]],
                                in_file,
                            )
                            sys.exit()
                        elif args.duplicates == "ignore":
                            if name == "text" and args.prefix_text_with_categories:
                                line = process_line(line, args.utt2category[i])
                            content.append(line)
                            if args.utt2category:
                                utt2cat.append(f"{parts[0]} {args.utt2category[i]}")
                        else:
                            print(
                                "- Duplicate key:",
                                parts[0],
                                " - line will not be appended",
                            )

        if content:
            print(len(content), name, end="\t")
            out_file = os.path.join(args.out_dir, name)
            with open(out_file, "w", encoding="utf-8") as fpw:
                fpw.write("\n".join(content) + "\n")
            print(out_file, "saved.")

        if args.utt2category:
            utt2cat_file = os.path.join(args.out_dir, "utt2category")
            if not os.path.exists(utt2cat_file):
                print(len(utt2cat), "utt2category", end="\t")
                with open(utt2cat_file, "w", encoding="utf-8") as fpw:
                    fpw.write("\n".join(utt2cat))
                print(utt2cat_file, "saved.")

                for soft in args.soft_links:
                    os.system(
                        "ln -svf "
                        + utt2cat_file
                        + " "
                        + os.path.join(args.out_dir, soft)
                    )

    print("\nRun utils/fix_data_dir.sh", args.out_dir)


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train_dirs",
        nargs="+",
        type=str,
        required=True,
        help="""path to different training / dev / test dirs to be merged
        (eg: dump/fbank_pitch/train_LANG1 dump/fbank_pitch/train_LANG2)""",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="""output dir where the merged files will be saved
        (eg: dump/fbank_pitch/train_LANG1_LANG2).""",
    )
    parser.add_argument(
        "--duplicates",
        type=str,
        default="error",
        choices=["error", "dedup", "ignore"],
        help="""what to do incase of duplicate keys (utt_ids).
        Throw error (or) de-duplicate i.e., keeps only one instance (or)
        ignore and copies everything which might throw error at a later point.""",
    )
    parser.add_argument(
        "--utt2category",
        default=[],
        nargs="+",
        type=str,
        help="""If category names are given, each train dir belongs to one category
        (one-to-one correspondence) (eg: LANG1 LANG2)""",
    )
    parser.add_argument(
        "--soft_links",
        default=["lid.scp", "utt2lang"],
        nargs="+",
        type=str,
        help="""Soft links from the given files to utt2category will be made
        (eg: lid.scp and utt2lang will be linked to utt2category)""",
    )
    parser.add_argument(
        "--prefix_text_with_categories",
        action="store_true",
        help="""every text utterance will be prefixed with the corresponding category name
        (eg: utt_ID [LANG1] word1 word2 .. will be each line text).
        This is useful when doing multilingual training with joint vocabulary and prompt-based
        decoding, where language ID is used as a prompt (prefix).""",
    )
    parser.add_argument(
        "--ignore_missing", action="store_true", help="ignore if some files are missing"
    )
    args = parser.parse_args()

    if len(args.utt2category) > 0:
        assert len(args.utt2category) == len(
            args.train_dirs
        ), "Number of train_dirs should be equal to number of categories in utt2category"

    return args


if __name__ == "__main__":
    main()
