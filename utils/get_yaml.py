#!/usr/bin/env python3
import argparse

import yaml


def get_parser():
    parser = argparse.ArgumentParser(
        description="get a specified attribute from a YAML file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inyaml")
    parser.add_argument(
        "-attrs", nargs="+", type=str, help='foo.bar will access yaml.load(inyaml)["foo"]["bar"]',
        default=["encoder_conf", "decoder_conf", "model_conf", "optim", "optim_conf", "bpemodel"]
    )
    parser.add_argument(
        "-extra", nargs="+", type=str,
        help="extra attributes apart from the default ones in -atts",
        default=[]
    )
    return parser


def main():
    args = get_parser().parse_args()
    with open(args.inyaml, "r") as f:
        indict = yaml.load(f, Loader=yaml.Loader)

    try:
        for attrib in args.attrs + args.extra:
            for attr in attrib.split("."):
                print("{:12s}:".format(attr), end=" ")
                if attr.isdigit():
                    attr = int(attr)
                if attr in indict:
                    print(indict[attr])
                else:
                    print("not found.")

    except KeyError:
        # print nothing
        # sys.exit(1)
        pass


if __name__ == "__main__":
    main()
