#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 03 Jun 2021
# Last modified : 03 Jun 2021

"""
Get total duration on utterances. If input is utt2dur, the calculation
is straightforward. If the input is wav.scp then will use ffprobe command
to get the duration of each recording.
"""

import os
import sys
import argparse
import subprocess


def main():
    """main method"""

    args = parse_arguments()

    subset_uttids = set()
    if args.subset_uttids:
        with open(args.subset_uttids, 'r', encoding='utf-8') as fpr:
            for line in fpr:
                uttid = line.strip().split()[0]
                subset_uttids.add(uttid)
        if args.verbose:
            print("Subset uttids:", len(subset_uttids))

    in_file = args.utt2dur if args.utt2dur else args.wavscp

    all_durs = []
    total_dur = 0.0
    with open(in_file, "r", encoding="utf-8") as fpr:
        lno = 0
        for line in fpr:
            lno += 1
            parts = line.strip().split()

            if subset_uttids:
                if parts[0] not in subset_uttids:
                    continue

            if args.utt2dur:
                if len(parts) != 2:
                    print(
                        "Each line should have two columns. Found:",
                        parts,
                        "at line",
                        lno,
                        file=sys.stderr,
                    )
                    sys.exit()
                total_dur += float(parts[1])

            elif args.wavscp:

                wav_fpath_index = args.wav_fpath_index
                # determine which col actually corresponds to audio file path
                if wav_fpath_index == -1:
                    for i, part in enumerate(parts):
                        if part.startswith("-"):
                            continue
                        else:
                            if os.path.isfile(part):
                                wav_fpath_index = i
                                break

                if wav_fpath_index == -1:
                    print(
                        "Could not determine which column actually corresponds to audio file.",
                        "Use --wav_path_index arg to pass it in command line. Col indices start from 0",
                    )
                    sys.exit()

                wav_fpath = parts[wav_fpath_index]

                assert os.path.isfile(
                    wav_fpath
                ), f"{wav_fpath} at col index {wav_fpath_index} is not a filepath. Line {line.strip()}"

                cmd = "ffprobe -v error -select_streams a:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 -i " + wav_fpath
                res = subprocess.run(cmd.split(), capture_output=True)
                dur = res.stdout.strip().decode("utf-8")
                all_durs.append(float(dur))
                total_dur += float(dur)

    print("{:6.1f} hrs".format(total_dur / 3600.0))


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    me_group = parser.add_mutually_exclusive_group(required=True)

    me_group.add_argument("-utt2dur", default="", help="path to utt2dur file")
    me_group.add_argument("-wavscp", default="", help="path to wav.scp file")
    parser.add_argument("-subset_uttids", help="path to a file where first col represent uttids that will considered to compute the duration")
    parser.add_argument(
        "-wav_fpath_index",
        default=-1,
        type=int,
        help="Col index in wav.scp that corresponds to actual audio file path. Column indices start from 0.",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
