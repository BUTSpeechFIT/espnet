#!/usr/bin/env python3
from espnet2.tasks.xlm import XLMTask


def get_parser():
    parser = XLMTask.get_parser()
    return parser


def main(cmd=None):
    """LM training.

    Example:

        % python xlm_train.py --print_config
        % python xlm_train.py --config conf/train_xlm.yaml
    """
    XLMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
