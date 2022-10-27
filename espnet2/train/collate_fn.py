from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.nets.pytorch_backend.nets_utils import pad_list


class CommonCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        assert check_argument_types()
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def common_collate_fn(
    collection: Collection[Tuple[str, Dict[str, Union[np.ndarray, str]]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[Union[List[str], Tuple[List, str]], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()

    uttids = []
    # uttids = [u for u, _ in collection]

    # check if dict has 'lid' and remove it from every dict-element in collection
    lid_for_batch = ""
    data = []
    for u, d in collection:
        uttids.append(u)
        if "lid" in d:
            if lid_for_batch:
                assert (
                    lid_for_batch == d["lid"]
                ), "All utterance IDs in a batch \
                    should belong to same language/lid/category. Found lids: \
                    {:s} and {:s}. Check utt ID {:s}. Use utt2category to indicate \
                    utt2lid mapping".format(
                    lid_for_batch, d["lid"], u
                )
            else:
                lid_for_batch = d["lid"]
            del d["lid"]

        data.append(d)

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:  # dict with keys: eg: speech, text

        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    if lid_for_batch:
        output = ((uttids, lid_for_batch), output)
    else:
        output = (uttids, output)

    assert check_return_type(output)
    return output
