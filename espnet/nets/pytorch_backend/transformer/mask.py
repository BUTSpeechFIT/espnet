# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Mask module."""

from typing import List

import torch


def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create mask for subsequent steps (size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


def target_mask(ys_in_pad, ignore_id):
    """Create mask for decoder self-attention.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int ignore_id: index of padding
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor (B, Lmax, Lmax)
    """
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    return ys_mask.unsqueeze(-2) & m


def mask_out(
    x: torch.Tensor, mask_probs: torch.Tensor, mlm_conf: dict
    ):
    """
    Decide of random words to mask out, and what target they get assigned.
    (taken from XLM original implementation)

    Args:
        x (torch.Tensor): Input batch of shape batch_size x seq_length
        mask_probs (torch.Tensor): 3 dim vector with probabilities for masking, keeping, replacement
        mlm_conf (dict): dictionary with information about vocab size, mask index and pad index
    Returns:
        [torch.Tensor, torch.Tensor, torch.Tensor]
    """

    device = x.device
    bs, slen = x.size()

    # need to select mlm_conf["word_pred"] % of tokens to mask
    # these should not be either sos_eos or ignore_id

    # nse -> not sos and eos indices
    nse = torch.where(x != mlm_conf["sos_and_eos_ix"])  # returns tuple
    # nsei -> not sos and eos and ignore_id indices
    nsei = torch.where(x[nse[0], nse[1]] != mlm_conf["ignore_id"])[0]  # returns vector

    for i in range(10):
        # random selection (eg: 15%) from nsei indices
        r_ixs = (torch.rand(len(nsei)) <= mlm_conf["word_pred"]).to(
            device=device
        )  # vector

        # map these random indices to the original x to create a mask matrix
        p_ixs = nse[0][nsei[r_ixs]], nse[1][nsei[r_ixs]]  # tuple
        pred_mask = torch.zeros_like(x).to(device=device, dtype=torch.bool)
        pred_mask[p_ixs] = 1

        # # define target words to predict
        # pred_mask = (torch.rand(bs, slen) <= mlm_conf["word_pred"]).to(device=device)

        # # do not predict ignore_id, sos_eos_ix
        # pred_mask[x == mlm_conf["ignore_id"]] = 0
        # pred_mask[x == mlm_conf["sos_and_eos_ix"]] = 0

        if pred_mask.sum() > 0:
            break

    assert pred_mask.sum() > 0, "Strange! Could not mask any of the tokens \
        in 10 random attempts. Are there too many very short utterances \
        in the batch. Debug the batch {:d} {:d}".format(bs, slen)

    # # mask a number of words == 0 [8] (faster with fp16)
    if mlm_conf["use_amp"]:
        pred_mask = pred_mask.view(-1)
        n1 = pred_mask.sum().item()
        n2 = max(n1 % 8, 8 * (n1 // 8))
        if n2 != n1:
            pred_mask[torch.nonzero(pred_mask).view(-1)[: n1 - n2]] = 0
        pred_mask = pred_mask.view(bs, slen)
        assert pred_mask.sum().item() % 8 == 0

    # generate possible targets / update x input
    _x_real = x[pred_mask]  # take the target tokens
    _x_rand = _x_real.clone().random_(mlm_conf["vocab_size"])
    _x_mask = _x_real.clone().fill_(mlm_conf["mask_index"])
    samples = torch.multinomial(mask_probs, len(_x_real), replacement=True)
    _x = (
        _x_mask * (samples == 0).long()
        + _x_real * (samples == 1).long()
        + _x_rand * (samples == 2).long()
    )
    x = x.masked_scatter(pred_mask, _x)

    assert 0 <= x.min() <= x.max() < mlm_conf["vocab_size"]
    assert x.size() == (bs, slen)
    assert pred_mask.size() == (bs, slen)

    return x, _x_real, pred_mask
