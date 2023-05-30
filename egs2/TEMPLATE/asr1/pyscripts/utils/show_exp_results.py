#!/usr/bin/env python3

import argparse
import glob
import json
import os
import sys
from pprint import pprint

CHRF_NAME = ""


def get_bleu(res_file):

    global CHRF_NAME
    res = {}
    with open(res_file, "r", encoding="utf-8") as fpr:
        _ = fpr.readline().strip()  # header
        data = ""
        for line in fpr:
            line = line.strip()
            if line.startswith("["):
                data += line + " "
            elif line.startswith("]"):
                data += line + " "
                break
            elif line.startswith("Case sensitive"):
                continue
            else:
                data += line + " "
        try:
            res = json.loads(data)
        except json.JSONDecodeError:
            print("Json decode err", res_file)

    assert res[0]["name"] == "BLEU", f"BLEU not found in {res_file}"
    assert res[1]["name"].startswith("chr"), f"chrF not found in {res_file}"

    CHRF_NAME = res[1]["name"].upper()

    if res:
        return res[0]["score"], res[1]["score"], res[0]["verbose_score"]
    else:
        return None, None, None


def plain_format(results):

    formatted_results = []
    for exp in results:
        formatted_results.append(f"+ {exp}")
        for dec in results[exp]:
            formatted_results.append(f"  - {dec}")
            for s, values in results[exp][dec].items():
                formatted_results.append(
                    "    . {:6s}  {:5.2f}  {:5.2f}  {:s}".format(
                        s, values[0], values[1], values[2]
                    )
                )
            formatted_results.append("    ")
    return formatted_results


def md_format(results):

    header = "| {:6s} | {:5s} | {:5s} | {:10s} |\n|---|---|---|---|".format(
        "SET", "BLEU", CHRF_NAME, "VERBOSE"
    )
    formatted_results = ["# RESULTS\n"]
    for exp in results:
        formatted_results.append(f"## {exp}\n")
        for dec in results[exp]:
            formatted_results.append(f"- {dec}\n")
            formatted_results.append(header)
            for s, values in results[exp][dec].items():
                formatted_results.append(
                    "| {:6s} | {:5.2f} | {:5.2f} | {:s} |".format(
                        s, values[0], values[1], values[2]
                    )
                )
            formatted_results.append("")
    return formatted_results


def main(args):
    """main method"""

    if args.metric == "bleu":
        sfx = "tc.txt"
    else:
        print(args.metric, "not implemeted yet")
        sys.exit()

    res_files = glob.glob(
        os.path.dirname(args.exp_base_dir)
        + f"/{args.pattern}/decode*/*/score_{args.metric}/result.{sfx}"
    )
    print(len(res_files))

    results = {}  # exp_dir: set_1: [], set_2: []

    for res_file in sorted(res_files):
        parts = os.path.dirname(res_file).split("/")
        # print(parts)
        set_name = parts[-2]
        decode_cfg = parts[-3]
        model_cfg = parts[-4]

        score1, score2, v_score = get_bleu(res_file)
        if score1 is None:
            continue

        if model_cfg not in results:
            results[model_cfg] = {}

        if decode_cfg not in results[model_cfg]:
            results[model_cfg][decode_cfg] = {}

        assert (
            set_name not in results[model_cfg][decode_cfg]
        ), f"{set_name} already in {results[model_cfg][decode_cfg]}"

        results[model_cfg][decode_cfg][set_name] = [score1, score2, v_score]

    # with open(args.out_file, "w", encoding="utf-8") as fpw:
    #    json.dump(results, fpw, indent=2)

    formatted_results = []

    if args.format == "plain":
        formatted_results = plain_format(results)
        if args.out_file:
            with open(args.out_file, "w", encoding="utf-8") as fpw:
                fpw.write("\n".join(formatted_results) + "\n")
        else:
            print("\n".join(formatted_results))

    elif args.format == "md":
        formatted_results = md_format(results)
        if args.out_file:
            with open(args.out_file, "w", encoding="utf-8") as fpw:
                fpw.write("\n".join(formatted_results) + "\n")
        else:
            print("\n".join(formatted_results))

    elif args.format == "json":
        formatted_results = results
        if args.out_file:
            with open(args.out_file, "w", encoding="utf-8") as fpw:
                json.dump(fpw, formatted_results)
        else:
            pprint(formatted_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--exp_base_dir",
        required=True,
        help="path to base dir where all the experiments are saved.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        required=True,
        choices=["wer", "cer", "bleu"],
        help="which metric to look",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        default=None,
        help="out file to store the results. If not given, results are printed on stdout",
    )
    parser.add_argument("-p", "--pattern", default="*", help="exp sub dir pattern")
    parser.add_argument("-f", "--format", default="md", choices=["md", "plain", "json"])
    args = parser.parse_args()

    main(args)
