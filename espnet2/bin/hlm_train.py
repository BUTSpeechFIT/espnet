#!/usr/bin/env python3
from espnet2.tasks.hlm_v1 import HLMTask


def get_parser():
    parser = HLMTask.get_parser()
    return parser


def main(cmd=None):
    """LM training.

    Example:

        % python hlm_train.py --print_config
        % python hlm_train.py --config conf/train_hlm.yaml
    """
    HLMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
