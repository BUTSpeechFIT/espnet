import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.lm.abs_model import AbsLM
from espnet2.lm.espnet_model import ESPnetMaskedLanguageModel

from espnet2.lm.transformer_xlm import TransformerXLM

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.mt.frontend.embedding import Embedding
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder

from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str_or_none


frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        embed=Embedding,
    ),
    type_check=AbsFrontend,
    default="embed",
)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=TransformerEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)


class XLMTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [frontend_choices, encoder_choices]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
        group = parser.add_argument_group(description="XLM related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]
        required += ["langs", "mlm_steps", "mlm_conf"]

        group.add_argument("--mlm_conf", type=dict)
        group.add_argument("--langs", type=str, help="list of languages. Eg: en,de,fr")
        group.add_argument(
            "--mlm_steps",
            type=str,
            help="MLM / TLM. For only MLM use en,de,fr. For only  TLM use en-de or de-fr. For MLM and TLM use en,de,en-de",
        )

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetMaskedLanguageModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word"],
            help="",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file fo sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

        assert check_return_type(parser)
        return parser

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                non_linguistic_symbols=args.non_linguistic_symbols,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("text",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetMaskedLanguageModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # get number of languages from args.langs
        # eg: en,pt -> 2 languages
        langs = args.langs.split(",")
        n_langs = len(langs)

        args.mlm_conf["vocab_size"] = vocab_size
        args.mlm_conf["use_amp"] = args.use_amp
        args.mlm_conf["ignore_id"] = args.model_conf["ignore_id"]
        args.mlm_conf["sos_and_eos_ix"] = vocab_size - 1
        try:
            args.mlm_conf["mask_index"] = token_list.index("<mask>")
        except ValueError:
            raise ValueError("Cannot find <mask> in the token_list")

        # 1. Build LM model

        args.frontend_conf['padding_idx'] = args.model_conf["ignore_id"]
        frontend = Embedding(input_size=vocab_size, **args.frontend_conf)
        input_size = frontend.output_size()

        encoder = TransformerEncoder(input_size=input_size, **args.encoder_conf)

        xlm = TransformerXLM(vocab_size, n_langs, args.mlm_conf, frontend, encoder)
        # 2. Build ESPnetModel
        # Assume the last-id is sos_and_eos
        model = ESPnetMaskedLanguageModel(
            lm=xlm, vocab_size=vocab_size, **args.model_conf
        )

        # FIXME(kamo): Should be done in model?
        # 3. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
