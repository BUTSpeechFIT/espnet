#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 03 Apr 2023
# Last modified : 03 Apr 2023

"""
Merge nbest hypotheses and logP scores into a json file
"""

import os
import sys
import argparse
import json
import glob


def load_key_and_score_file(fname, verbose):
    """Load utt IDs, hypotheses and logP scores"""

    data = {}
    # utt ids in the current file that do not have any hypotheses generated
    no_hyp = set()
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            try:
                uttid, text = line.split(" ", maxsplit=1)
            except ValueError as err:
                if verbose:
                    print(fname, "|", str(err), "|", line)
                no_hyp.add(line.strip())
                continue

            if uttid in data:
                print("Duplicate", uttid, "in", fname)
                sys.exit()
            else:
                data[uttid] = [text]

    score_file = os.path.join(os.path.dirname(fname), 'score')
    with open(score_file, 'r', encoding='utf-8') as fpr:
        for line in fpr:
            uttid, score = line.strip().split()
            score = score.replace("tensor(", "")
            log_p = float(score.replace(")", ""))

            if uttid not in no_hyp:
                data[uttid].append(log_p)

    return data


def main():
    """main method"""

    args = parse_arguments()

    in_files = sorted(glob.glob(args.decode_log_dir + "/output.*/*best_recog/text"))

    print("Found", len(in_files), "files to merge.")

    nbest = {}
    for fname in in_files:

        n = int(fname.split("/")[-2].replace("best_recog", ""))
        data = load_key_and_score_file(fname, args.verbose)

        for uid, text in data.items():
            if uid not in nbest:
                nbest[uid] = {}
            if n not in nbest[uid]:
                nbest[uid][n] = text
            else:
                print("strange:", n, "already in nbest[uid]", uid)
                sys.exit()

    with open(args.out_json_file, "w", encoding="utf-8") as fpw:
        json.dump(nbest, fpw, indent=2, ensure_ascii=False)
    print(args.out_json_file, 'saved.', len(nbest), 'utt ids')


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("decode_log_dir", help="path to decode log dir containing output.x sub dirs")
    parser.add_argument("out_json_file", help="out json file to save the nbest hypotheses with logP scores")
    parser.add_argument("--verbose", action="store_true", help="show utterance ID without any hypotheses")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
