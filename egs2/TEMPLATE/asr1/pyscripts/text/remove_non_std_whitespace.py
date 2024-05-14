#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Remove non-standard whitespace in text.
"""

import os
import sys
import argparse
import re


def main():
    """main method"""

    args = parse_arguments()

    patt = re.compile(r"\xa0")  # you may add additional unicode characters

    orig = []
    lines = []
    with open(args.in_file, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            orig.append(line)
            if re.search(patt, line):
                # print(line)
                lines.append(re.sub(patt, "", line))
            else:
                lines.append(line)

    assert len(orig) == len(lines)

    if not os.path.exists(args.in_file + ".bak"):
        os.system("cp -v " + args.in_file + " " + args.in_file + ".bak")
    else:
        print(args.in_file + ".bak already exists.")
        sys.exit()

    with open(args.in_file, "w", encoding="utf-8") as fpw:
        fpw.write("\n".join(lines) + "\n")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("in_file", help="path to data/train/text or similar")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
