from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import mask_out

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class TransformerHLMv2(AbsESPnetModel):
    """Transformer Encoder-Decoder based hybrid MT and MLM model

    This is taken from espnet2/mt/espnet_model.py and further modified
    """

    def __init__(
        self,
        vocab_size: dict,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        mlm_conf: dict,
        src_vocab_size: int = 0,
        src_token_list: Union[Tuple[str, ...], List[str]] = [],
        lid2int: dict = None,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        mlm_weight: float = 1.0,
        length_normalized_loss: bool = False,
        report_bleu: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_mask: str = "<mask>",
        extract_feats_in_collect_stats: bool = True,
        share_decoder_input_output_embed: bool = False,
        share_encoder_decoder_input_embed: bool = False,
    ):
        """
        Args:
            vocab_size (dict): {lid1: vocab_size, lid2: vocab_size ...}
            token_list (Tuple or List):
            frontend:
            encoder:

        """
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.src_vocab_size = src_vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.lid2int = lid2int
        self.mlm_conf = mlm_conf
        self.mlm_conf["ignore_id"] = ignore_id
        self.mlm_conf["mask_index"] = src_token_list.index(sym_mask)
        self.mlm_conf["vocab_size"] = self.src_vocab_size
        self.mlm_conf["sos_and_eos_ix"] = self.sos

        self.pred_probs = torch.FloatTensor(
            [mlm_conf["word_mask"], mlm_conf["word_keep"], mlm_conf["word_rand"],]
        )
        self.mlm_weight = mlm_weight

        if share_decoder_input_output_embed:
            if decoder.output_layer is not None:
                decoder.output_layer.weight = decoder.embed[0].weight
                logging.info(
                    "Decoder input embedding and output linear layer are shared"
                )
            else:
                logging.warning(
                    "Decoder has no output layer, so it cannot be shared "
                    "with input embedding"
                )

        if share_encoder_decoder_input_embed:
            if src_vocab_size == vocab_size:
                frontend.embed[0].weight = decoder.embed[0].weight
                logging.info("Encoder and decoder input embeddings are shared")
            else:
                logging.warning(
                    f"src_vocab_size ({src_vocab_size}) does not equal tgt_vocab_size"
                    f" ({vocab_size}), so the encoder and decoder input embeddings "
                    "cannot be shared"
                )

        self.frontend = frontend
        self.lang_embeddings = torch.nn.Embedding(
            len(self.lid2int), self.frontend.output_size()
        )
        # self.preencoder = preencoder
        # self.postencoder = postencoder
        self.encoder = encoder
        self.decoder = decoder

        self.criterion_mt = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.encoder_pred_layer = torch.nn.Linear(
            self.frontend.output_size(), vocab_size
        )
        # share encoder input embeddings and pred layer weights
        self.encoder_pred_layer.weight = frontend.embed[0].weight
        self.encoder_log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.neg_log_likelihood = torch.nn.NLLLoss(reduction="mean")

        # MT error calculator
        if report_bleu:
            self.mt_error_calculator = MTErrorCalculator(
                token_list, sym_space, sym_blank, report_bleu
            )
        else:
            self.mt_error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            text: (Batch, Length) Source or Target (L1 or L2) language text token IDs
            text_lengths: (Batch,) Source or Target (L1 or L2) language text lengths
            src_text: (Batch, length) Source (L1) language text token IDs
            src_text_lengths: (Batch,) Source (L1) language text lengths
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            text.shape[0]
            == text_lengths.shape[0]
            == src_text.shape[0]
            == src_text_lengths.shape[0]
        ), (text.shape, text_lengths.shape, src_text.shape, src_text_lengths.shape)

        self.pred_probs = self.pred_probs.to(device=text.device)

        import ipdb

        ipdb.set_trace()

        batch_size = src_text.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        src_text = src_text[:, : src_text_lengths.max()]

        # 1. Encoder
        (
            encoder_out,
            encoder_out_lens,
            src_text_targets,
            src_text_pred_mask,
        ) = self.encode(src_text, src_text_lengths)

        # 2a. Encoder MLM loss
        mlm_loss, mlm_acc = self._calc_encoder_mlm_loss(
            encoder_out, src_text_targets, src_text_pred_mask
        )

        # 2a. Attention-decoder branch (MT)
        loss_mt_att, acc_mt_att, bleu_mt_att = self._calc_mt_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # 3. Loss computation
        loss = loss_mt_att + (self.mlm_weight * mlm_loss)

        stats = dict(
            loss=loss.detach(),
            mt_loss=loss_mt_att.detach(),
            mlm_loss=mlm_loss.detach(),
            mt_acc=acc_mt_att,
            mlm_acc=mlm_acc,
            bleu=bleu_mt_att,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(src_text, src_text_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = src_text, src_text_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        is_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by mt_inference.py

        Args:
            src_text: (Batch, Length, ...)
            src_text_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            (
                feats,
                feats_lengths,
                src_text_targets,
                src_text_pred_mask,
            ) = self._extract_feats(src_text, src_text_lengths, is_inference)

            # 2. Data augmentation
            # if self.specaug is not None and self.training:
            #     feats, feats_lengths = self.specaug(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        # if self.preencoder is not None:
        #    feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        # Post-encoder, e.g. NLU
        # if self.postencoder is not None:
        #   encoder_out, encoder_out_lens = self.postencoder(
        #       encoder_out, encoder_out_lens
        #    )

        assert encoder_out.size(0) == src_text.size(0), (
            encoder_out.size(),
            src_text.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens, src_text_targets, src_text_pred_mask

    def _extract_feats(
        self,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        is_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert src_text_lengths.dim() == 1, src_text_lengths.shape

        # for data-parallel
        src_text = src_text[:, : src_text_lengths.max()]
        src_text, _ = add_sos_eos(src_text, self.sos, self.eos, self.ignore_id)
        src_text_lengths = src_text_lengths + 1

        src_text_targets = None
        src_text_pred_mask = None
        if not is_inference:
            # Masking in source text
            src_text, src_text_targets, src_text_pred_mask = mask_out(
                src_text, self.pred_probs, self.mlm_conf
            )

        if self.frontend is not None:
            # Frontend
            #  e.g. Embedding Lookup
            # src_text (Batch, NSamples) -> feats: (Batch, NSamples, Dim)
            feats, feats_lengths = self.frontend(src_text, src_text_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = src_text, src_text_lengths
        return feats, feats_lengths, src_text_targets, src_text_pred_mask

    def _calc_mt_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_mt(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute bleu using attention-decoder
        if self.training or self.mt_error_calculator is None:
            bleu_att = None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            bleu_att = self.mt_error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, bleu_att

    def _calc_encoder_mlm_loss(
        self,
        encoder_out: torch.Tensor,
        src_text_targets: torch.Tensor,
        src_target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate negative log-likelihood of predictions from
        the output of the encoder, whose targets are the masked inputs.

        Args:
            encoder_out (torch.Tensor): (bs x slen) Output representations from the encoder
            src_text_targets (torch.Tensor): vector of target IDs
            src_target_mask (torch.Tensor): (bs x slen) Positions in the encoder_out that correspond to the target IDs

        Returns:
            torch.Tensor: cross-entropy loss
            torch.Tensor: MLM accuracy
        """

        pred_logits = self.encoder_pred_layer(encoder_out)
        pred_log_probs = self.encoder_log_softmax(pred_logits)

        t_log_probs = pred_log_probs[src_target_mask]

        nll = self.neg_log_likelihood(
            t_log_probs.view(-1, t_log_probs.shape[-1]), src_text_targets.view(-1)
        )

        t_preds = torch.argmax(t_log_probs, dim=1)
        acc = (t_preds == src_text_targets).sum() / len(src_text_targets)

        return nll, acc