#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 01 Dec 2022
# Last modified : 01 Dec 2022

"""
"""

import os
import sys
import argparse

BASE_NAMES = ["tokens.txt", "bpe.model"]


def main():
    """main method"""

    args = parse_arguments()

    flag = False
    data = {}
    for name in BASE_NAMES:
        data[name] = []
        for i, tok_dir in enumerate(args.token_listdirs):
            fname = os.path.realpath(os.path.join(tok_dir, name))
            if not os.path.exists(fname):
                print("- File not found:", fname)
                flag = True
            else:
                data[name].append(f"{args.categories[i]} {fname}")

    if flag:
        print("- One or more required files are not found.")
        sys.exit()

    else:
        os.makedirs(args.out_token_listdir, exist_ok=True)
        for key, lst in data.items():
            print("\n".join(lst))
            out_file = os.path.join(args.out_token_listdir, key)
            with open(out_file, "w", encoding="utf-8") as fpw:
                fpw.write("\n".join(lst) + "\n")
            print(out_file, "saved.")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-token_listdirs",
        nargs="+",
        required=True,
        type=str,
        help="input base token_listdir",
    )
    parser.add_argument(
        "-categories",
        nargs="+",
        required=True,
        type=str,
        help="list of categories or languages. should be same number as list of token_listdirs",
    )
    parser.add_argument(
        "-out_token_listdir", required=True, type=str, help="output token_listdir"
    )

    args = parser.parse_args()

    assert len(args.categories) == len(
        args.token_listdirs
    ), "num of categories != num of token_listdirs"

    return args


if __name__ == "__main__":
    main()
