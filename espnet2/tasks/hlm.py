import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.samplers.build_batch_sampler import build_batch_sampler
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory

from espnet2.train.dataset import ESPnetDataset

from espnet2.asr.decoder.abs_decoder import AbsDecoder

# from espnet2.asr.decoder.rnn_decoder import RNNDecoder
# from espnet2.asr.decoder.transformer_decoder import (
#    DynamicConvolution2DTransformerDecoder,  # noqa: H301
# )
# from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
# from espnet2.asr.decoder.transformer_decoder import (
#    LightweightConvolution2DTransformerDecoder,  # noqa: H301
# )
# from espnet2.asr.decoder.transformer_decoder import (
#    LightweightConvolutionTransformerDecoder,  # noqa: H301
# )
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder

# from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
# from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder

# from espnet2.asr.encoder.contextual_block_transformer_encoder import (
#    ContextualBlockTransformerEncoder,  # noqa: H301
# )
# from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.lm.transformer_hlm import TransformerHLMv2
from espnet2.lm.transformer_hlm_v1 import TransformerHLM
from espnet2.mt.frontend.embedding import EmbeddingDict
from espnet2.tasks.abs_task import AbsTask, IteratorOptions
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import MutliTokenizerCommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        embed=EmbeddingDict,
    ),
    type_check=AbsFrontend,
    default="embed",
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        # conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        # contextual_block_transformer=ContextualBlockTransformerEncoder,
        # vgg_rnn=VGGRNNEncoder,
        # rnn=RNNEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        # lightweight_conv=LightweightConvolutionTransformerDecoder,
        # lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        # dynamic_conv=DynamicConvolutionTransformerDecoder,
        # dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        # rnn=RNNDecoder,
    ),
    type_check=AbsDecoder,
    default="transformer",
)


class HLMTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --preencoder and --preencoder_conf
        # preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        # postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_flist"]
        required += ["langs"]

        group.add_argument(
            "--langs",
            type=str,
            help="List of languages. (eg: en,te,fr,hi,it,ru,zh,taq)",
        )
        group.add_argument(
            "--token_flist",
            type=str_or_none,
            default=None,
            help="An flist containing lang ID to tokens.txt file path where tokens.txt is a text file with int-id to token mapping (for each language)",
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
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(TransformerHLM),
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
            choices=["bpe", "char", "word", "phn"],
            help="The target text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel_flist",
            type=str_or_none,
            default=None,
            help="File list containing lid to the model file of sentencepiece (for each language)",
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

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = MutliTokenizerCommonPreprocessor(
                train=train,
                token_type=[args.token_type, args.src_token_type],
                token_list=[args.token_list, args.src_token_list],
                bpemodel=[args.bpemodel, args.src_bpemodel],
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                text_name=["text", "src_text"],
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("src_text", "text")
        else:
            # Recognition mode
            retval = ("src_text",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ()
        else:
            retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_task_iter_factory(
        cls, args: argparse.Namespace, iter_options: IteratorOptions, mode: str
    ) -> AbsIterFactory:
        """Task specific iter factory"""

        dataset = ESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
            preserve_lid=True,
            lid2int=args.lid2int,
        )
        cls.check_task_requirements(
            dataset, args.allow_variable_data_keys, train=iter_options.train
        )

        if Path(
            Path(iter_options.data_path_and_name_and_type[0][0]).parent, "utt2category"
        ).exists():
            utt2category_file = str(
                Path(
                    Path(iter_options.data_path_and_name_and_type[0][0]).parent,
                    "utt2category",
                )
            )
        else:
            utt2category_file = None

        batch_sampler = build_batch_sampler(
            type=iter_options.batch_type,
            shape_files=iter_options.shape_files,
            fold_lengths=args.fold_length,
            batch_size=iter_options.batch_size,
            batch_bins=iter_options.batch_bins,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
            drop_last=False,
            min_batch_size=torch.distributed.get_world_size()
            if iter_options.distributed
            else 1,
            utt2category_file=utt2category_file,
        )

        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]

        bs_list = [len(batch) for batch in batches]

        logging.info(f"[{mode}] dataset:\n{dataset}")
        logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
        logging.info(
            f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
            f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
        )

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        return SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
        )

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> TransformerHLM:
        """Build HLM """

        assert check_argument_types()

        if isinstance(args.token_list, str):
            fnames = args.token_list.split(",")
            for fname in fnames:
                with open(fname, encoding="utf-8") as f:
                    token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        args.lid2int = {}
        for lid in args.langs.split(","):
            assert lid not in args.lid2int, f"Duplicate {lid} -> {args.langs}"
            args.lid2int[lid] = len(args.lid2int)

        args.model_conf["mlm_conf"]["use_amp"] = args.use_amp

        if args.src_token_list is not None:
            if isinstance(args.src_token_list, str):
                with open(args.src_token_list, encoding="utf-8") as f:
                    src_token_list = [line.rstrip() for line in f]

                # Overwriting src_token_list to keep it as "portable".
                args.src_token_list = list(src_token_list)
            elif isinstance(args.src_token_list, (tuple, list)):
                src_token_list = list(args.src_token_list)
            else:
                raise RuntimeError("token_list must be str or list")
            src_vocab_size = len(src_token_list)
            logging.info(f"Source vocabulary size: {src_vocab_size }")
        else:
            src_token_list, src_vocab_size = None, None

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(input_size=src_vocab_size, **args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 3. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            **args.decoder_conf,
        )

        # 8. Build model
        model = TransformerHLM(
            vocab_size=vocab_size,
            src_vocab_size=src_vocab_size,
            frontend=frontend,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            token_list=token_list,
            src_token_list=src_token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
