#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding Frontend for text based inputs."""

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet2.asr.frontend.abs_frontend import AbsFrontend
import torch
from typeguard import check_argument_types
from typing import Tuple, Union


class Embedding(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
        padding_idx: Union[None, int] = None,
        use_emb_norm: bool = False,
        emb_dropout: float = 0.0,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
            padding_idx (None | int): index for padding / <blank> token
            use_emb_norm: Apply LayerNorm after emb + pos_emb
            emb_dropout: Apply dropout after LN(emb + pos_emb) or emb + pos_emb
        """
        assert check_argument_types()
        super().__init__()
        self.embed_dim = embed_dim

        if use_emb_norm:
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, embed_dim, padding_idx=padding_idx),
                pos_enc_class(embed_dim, positional_dropout_rate),
                torch.nn.LayerNorm(embed_dim),
                torch.nn.Dropout(emb_dropout),
            )
        else:
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, embed_dim, padding_idx=padding_idx),
                pos_enc_class(embed_dim, positional_dropout_rate),
                torch.nn.Dropout(emb_dropout),
            )

    def forward(
        self, input: torch.Tensor, input_lengths: Union[None, torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T) or (B, T,D), with D.
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D).
            Tensor: Output lengths within batch.
        """
        x = self.embed(input)

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim


class EmbeddingDict(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    def __init__(
        self,
        lid2vocab_size: dict,
        embed_dim: int = 512,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
        padding_idx: Union[None, int] = None,
        use_emb_norm: bool = False,
        emb_dropout: float = 0.0,
    ):
        """Initialize.

        Args:
            lid2vocab_size: Lang ID to vocab size mapping
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
            padding_idx (None | int): index for padding / <blank> token
            use_emb_norm: Apply LayerNorm after emb + pos_emb
            emb_dropout: Apply dropout after LN(emb + pos_emb) or emb + pos_emb
        """
        assert check_argument_types()
        super().__init__()
        self.embed_dim = embed_dim

        self.embed = torch.nn.ModuleDict()

        for lid in lid2vocab_size:
            if use_emb_norm:
                self.embed[lid] = torch.nn.Sequential(
                    torch.nn.Embedding(input_size, embed_dim, padding_idx=padding_idx),
                    pos_enc_class(embed_dim, positional_dropout_rate),
                    torch.nn.LayerNorm(embed_dim),
                    torch.nn.Dropout(emb_dropout),
                )
            else:
                self.embed[lid] = torch.nn.Sequential(
                    torch.nn.Embedding(input_size, embed_dim, padding_idx=padding_idx),
                    pos_enc_class(embed_dim, positional_dropout_rate),
                    torch.nn.Dropout(emb_dropout),
            )

    def forward(
            self,
            lids: torch.Tensor,
            x: torch.Tensor,
            x_lengths: Union[None, torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            lids: Lang IDs of the input x (B, )
            x: Input (B, T) or (B, T,D), with D.
            x_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D).
            Tensor: Output lengths within batch.
        """
        embs = []
        for i, lid in enumerate(lids):
            embs.append(self.embed_dict[lid](x[i]))
        embs = torch.cat(embs)

        return embs, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim
