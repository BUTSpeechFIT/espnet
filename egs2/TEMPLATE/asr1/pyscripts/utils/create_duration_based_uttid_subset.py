#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju


"""
Create a subset of utt ID based on duration
"""

import os
import yaml
import argparse
from random import shuffle
import numpy as np


def load_keys(fname):

    utt_ids = []
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            parts = line.strip().split()
            utt_ids.append(parts[0])
    return utt_ids


def print_dur(dur_s, prefix=""):

    print(
        "{:s} {:.2f} s ({:.1f} m) ({:.1f} h)".format(
            prefix, dur_s, dur_s / 60.0, dur_s / 3600.0
        )
    )


def main():
    """main"""

    args = parse_arguments()

    target_dur_s = 0.0
    if args.target_dur_units == "h":
        target_dur_s = args.target_dur * 3600.0
    elif args.target_dur_units == "m":
        target_dur_s = args.target_dur * 60.0
    else:
        target_dur_s = args.target_dur

    print_dur(target_dur_s, "- Desired target duration:")

    utts = load_keys(args.utt2dur)
    durs = np.loadtxt(args.utt2dur, dtype=float, usecols=[1])
    print("- Loaded utt IDs", len(utts))

    utt2dur = {}
    for i, u in enumerate(utts):
        utt2dur[u] = durs[i]

    utts2consider = set()

    incl = []
    incl_dur = 0.0
    if args.include:
        incl = load_keys(args.include)
        print("- Mandatory include utt IDs (subject to target duration):", len(incl))
        for u in incl:
            incl_dur += utt2dur[u]
        print_dur(incl_dur, "  . Duration:")

    dur_excl = 0.0
    excl = []
    if args.exclude:
        excl = load_keys(args.utt2dur)
        print("- Mandatory exclude utt IDs:", len(excl))
        if incl:
            assert (
                len(set(incl) & set(excl)) == 0
            ), "Include utt IDs and exclude utt IDs are not disjoint."

        utts2consider = set(utts) - set(excl)
        for u in utts2consider:
            dur_excl += utt2dur[u]
        print_dur(dur_excl, "  . Duration:")

    else:
        utts2consider = set(utts) - set(incl)

    assert (
        durs.sum() - dur_excl >= args.target_dur
    ), f"Cannot achieve target duration {args.target_dur}, as total duration (after exclusion) is {durs.sum() - dur_excl}"

    if args.sel == "rand":
        shuffle(utts)
        shuffle(incl)

    for u in utts2consider:
        print(
            "\r- Included utts {:9d} duration: {:.2f} s".format(len(incl), incl_dur),
            end=" ",
        )
        if incl_dur > target_dur_s:
            break

        incl.append(u)
        incl_dur += utt2dur[u]

    print()

    if incl_dur > target_dur_s:
        print("- Removing some utts to meet the target")
        while incl_dur > target_dur_s:
            u = incl.pop(0)
            incl_dur -= utt2dur[u]

    print("- Final selected utts:", len(incl))
    print_dur(incl_dur, "  . Duration:")

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    np.savetxt(args.out_file, sorted(incl), fmt="%s")
    print("-", args.out_file, "saved.")


def parse_arguments():
    """parse command line args"""

    parser = argparse.ArgumentParser()
    parser.add_argument("utt2dur", help="utt 2 dur file")
    parser.add_argument("target_dur", type=float, help="target duration")
    parser.add_argument(
        "target_dur_units",
        type=str,
        choices=["h", "m", "s"],
        help="target duratio in hours or minutes or seconds",
    )
    parser.add_argument("out_file", help="out file to save te utt ids")
    parser.add_argument(
        "--sel", choices=["seq", "rand"], default="seq", help="how to select utt ids"
    )
    parser.add_argument(
        "-i",
        "--include",
        default=None,
        help="path to file with utt ids that shall be included and taken account while computing target duration",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        default=None,
        help="path to file with utt ids that shall be excluded",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
