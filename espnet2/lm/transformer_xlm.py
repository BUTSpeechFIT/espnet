from typing import Any, List, Tuple

import torch
import torch.nn as nn

from espnet2.lm.abs_model import AbsLM
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend

from espnet.nets.pytorch_backend.transformer.mask import mask_out

"""
Transformer XLM - masked LM
"""


class TransformerXLM(AbsLM):
    """Transformer Cross-lingal Masked/Translation Language Model"""

    def __init__(
        self,
        vocab_size: int,
        n_langs: int,
        mlm_conf: dict,
        frontend: AbsFrontend,
        encoder: AbsEncoder,
    ):
        """
        Args:
            vocab_size (int):
            n_langs (int):
            mlm_conf (dict):
            frontend (AbsFronted):
            encoder (AbsEncoder):
        """
        super().__init__()

        self.n_langs = n_langs
        self.mlm_conf = mlm_conf

        self.pred_probs = torch.FloatTensor(
            [mlm_conf["word_mask"], mlm_conf["word_keep"], mlm_conf["word_rand"],]
        )

        self.embed = frontend
        self.encoder = encoder
        self.decoder = nn.Linear(self.embed.output_size(), vocab_size)

    def _target_mask(self, x):
        x, _x_real, pred_mask = mask_out(x, self.pred_probs, self.mlm_conf)
        return x, _x_real, pred_mask

    def forward(
        self, inp: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            ilens (torch.Tensor): Input lengths. (batch,)

        """

        self.pred_probs = self.pred_probs.to(device=inp.device)
        x, _x_real, pred_mask = self._target_mask(inp)

        x, _ = self.embed(x)
        h, _, _ = self.encoder(x, ilens)
        y = self.decoder(h)

        return y, _x_real, pred_mask

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        y = y.unsqueeze(0)
        h, _, cache = self.encoder.forward_one_step(
            self.embed(y), self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, vocab_size)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            self.embed(ys), self._target_mask(ys), cache=batch_state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
