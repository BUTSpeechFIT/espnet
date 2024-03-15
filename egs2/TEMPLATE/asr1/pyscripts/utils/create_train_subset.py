#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 26 Jul 2022
# Last modified : 26 Jul 2022

"""
Create a training subset dir given the subset utt ids and the original training data dir
"""

import os
import yaml
import argparse
import glob
from shutil import copyfile


def write_simple_flist(some_list, out_fname):
    """Write the elements in the list line by line
    in the given out file

    Parameters:
    -----------
    some_list (list): list of elements
    out_fname (str): output file name

    """

    with open(out_fname, "w", encoding="utf-8") as fpw:
        fpw.write("\n".join(some_list))


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

    assert os.path.exists(args.utt_ids_file), f"{args.utt_ids_file} not found."

    os.makedirs(args.out_dir, exist_ok=True)
    fnames = glob.glob(args.in_dir + "/*")
    print("- Found {:d} files in input dir.".format(len(fnames)))
    print("  - ", [os.path.basename(f) for f in fnames])
    utt_ids = set()
    if args.utt_ids_file.rsplit(".", 1)[-1] == "yaml":
        print("This is very specific to MUST C corpus")
        data = {}
        with open(args.utt_ids_file, "r") as fpr:
            data = yaml.load(fpr, Loader=yaml.FullLoader)
        for i, row in enumerate(data):
            print("\r{:5d}".format(i + 1), end=" ")
            rec_id = int(row["wav"].rsplit(".", 1)[0].split("_")[-1])
            offset = round(float(row["offset"] * 1000.0))
            duration = round(float(row["duration"] * 1000.0))
            utt_id = "ted_{:05d}_{:07d}_{:07d}".format(
                rec_id, offset, offset + duration
            )
            if utt_id in utt_ids:
                print("- Strange", utt_id, "already present", row)
            utt_ids.add(utt_id)
        print()
    else:
        utt_ids = set(load_keys(args.utt_ids_file))
    print("- Loaded {:d} target utt ids".format(len(utt_ids)))

    utt2spk_f = os.path.join(args.in_dir, "utt2spk")
    spk_ids = set()
    if os.path.exists(utt2spk_f):
        print("  - Getting spk information from utt2spk")
        with open(utt2spk_f, "r", encoding="utf-8") as fpr:
            for line in fpr:
                parts = line.strip().split()
                if parts[0] in utt_ids:
                    spk_ids.add(parts[-1])

    print("  - Found {:d} spk ids".format(len(spk_ids)))

    print("- Creating files in destination dir ..")
    for i, fname in enumerate(fnames):
        if not os.path.isfile(fname):
            continue

        base = os.path.basename(fname).rsplit(".", 1)[0]
        subset = []
        with open(fname, "r", encoding="utf-8") as fpr:
            for line in fpr:
                line = line.strip()
                parts = line.split(" ", maxsplit=1)
                if base in {"cmvn", "wav", "spk2utt", "spk_map"}:
                    if parts[0] in spk_ids:
                        subset.append(line)
                    elif parts[0] in utt_ids:
                        subset.append(line)
                elif base in ("feats_type", "feats_dim"):
                    subset.append(line)
                else:
                    if parts[0] in utt_ids:
                        subset.append(line)

        print(
            "  {:2d} {:>15s} {:>6d}".format(
                i + 1, os.path.basename(fname), len(subset)
            ),
            end=" ",
        )
        if subset:
            print("\u2713")
            write_simple_flist(
                subset, os.path.join(args.out_dir, os.path.basename(fname))
            )
        else:
            print("\u2713")
            copyfile(fname, os.path.join(args.out_dir, os.path.basename(fname)))


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("in_dir", help="original training data dir")
    parser.add_argument("utt_ids_file", help="file with utt ids. This will be .yaml file for MUSTC corpus.")
    parser.add_argument("out_dir", help="target output dir to save the subset files")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
