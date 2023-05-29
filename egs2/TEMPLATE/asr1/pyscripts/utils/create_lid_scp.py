#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 16 Feb 2023
# Last modified : 16 Feb 2023

"""
Creates lid.scp, utt2category and utt2lang files (all three are identical), given a file with utt ID and target langauge ID
"""

import os
import argparse
from shutil import copyfile


def load_keys(fname):

    utt_ids = []
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            parts = line.strip().split()
            utt_ids.append(parts[0])
    return utt_ids


def main():
    """main method"""

    args = parse_arguments()

    utt_ids = load_keys(args.in_file)
    print("Loaded utt IDs:", len(utt_ids))

    lid = args.lid
    out_dir = os.path.dirname(args.in_file)
    utt2cat_file = os.path.join(out_dir, "utt2category")
    with open(utt2cat_file, "w", encoding="utf-8") as fpw:
        for i, uid in enumerate(utt_ids):
            fpw.write(f"{uid} {lid}\n")
    copyfile(utt2cat_file, os.path.join(out_dir, "utt2lang"))
    copyfile(utt2cat_file, os.path.join(out_dir, "lid.scp"))


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "in_file",
        help="A input file with two columns, where first column should be uttID. Eg: text or utt2spk",
    )
    parser.add_argument("lid", help="target lang ID")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
