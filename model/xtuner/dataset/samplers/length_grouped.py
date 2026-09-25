# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Optional, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.logging import print_log
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Sampler


def get_length_grouped_indices(lengths, group_batch_size, generator=None):
    def process(lengths, group_batch_size, generator=None):
        indices = torch.randperm(len(lengths), generator=generator)
        megabatches = [
            indices[i : i + group_batch_size].tolist()
            for i in range(0, len(lengths), group_batch_size)
        ]
        megabatches = [
            sorted(megabatch, key=lambda i: lengths[i], reverse=True)
            for megabatch in megabatches
        ]
        return megabatches

    assert all(leng != 0 for leng in lengths), "Should not have zero length."
    if all(leng > 0 for leng in lengths) or all(leng < 0 for leng in lengths):
        # all samples are in the same modality
        megabatches = process(lengths, group_batch_size, generator=generator)
    else:
        mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
        lang_indices, lang_lengths = zip(
            *[(i, -l) for i, l in enumerate(lengths) if l < 0]
        )
        mm_megabatches = []
        for mm_megabatch in process(mm_lengths, group_batch_size, generator=generator):
            mm_megabatches.append([mm_indices[i] for i in mm_megabatch])
        lang_megabatches = []
        for lang_megabatch in process(
            lang_lengths, group_batch_size, generator=generator
        ):
            lang_megabatches.append([lang_indices[i] for i in lang_megabatch])

        last_mm = mm_megabatches[-1]
        last_lang = lang_megabatches[-1]
        last_batch = last_mm + last_lang
        megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]

        megabatch_indices = torch.randperm(len(megabatches), generator=generator)
        megabatches = [megabatches[i] for i in megabatch_indices]

        if len(last_batch) > 0:
            megabatches.append(
                sorted(last_batch, key=lambda i: abs(lengths[i]), reverse=True)
            )

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length,
    # the longest element is the first
    megabatch_maximums = [abs(lengths[megabatch[0]]) for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = (
        megabatches[max_idx][0],
        megabatches[0][0],
    )

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        dataset: Sized,
        per_device_batch_size: int,
        length_property="length",
        mega_batch_mult: Optional[int] = None,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        print_log("LengthGroupedSampler is used.", logger="current")
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            num_iters = math.ceil(
                len(self.dataset) / world_size / per_device_batch_size
            )
            self.num_samples = num_iters * per_device_batch_size
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

        total_batch_size = per_device_batch_size * self.world_size
        if mega_batch_mult is None:
            # Default for mega_batch_mult: 50 or the number to get 4
            # megabatches, whichever is smaller.
            mega_batch_mult = min(len(self.dataset) // (total_batch_size * 4), 50)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1
        self.group_batch_size = mega_batch_mult * total_batch_size

        if isinstance(self.dataset, TorchConcatDataset):
            length = []
            for sub_dataset in self.dataset.datasets:
                length.extend(getattr(sub_dataset, length_property))
            self.length = length
        else:
            self.length = getattr(self.dataset, length_property)
        assert isinstance(self.length, (list, tuple))

        self.total_batch_size = total_batch_size
        print_log(
            f"LengthGroupedSampler construction is complete, "
            f"and the selected attribute is {length_property}",
            logger="current",
        )

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(
            lengths=self.length,
            group_batch_size=self.group_batch_size,
            generator=generator,
        )
        assert len(set(indices)) == len(indices)
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[
                : self.total_size
            ]
        # subsample
        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


import math

class BalancedLengthGroupedSamplerSingleGPU(LengthGroupedSampler):
    def __init__(
        self,
        dataset: Sized,
        per_device_batch_size: int,
        length_property="length",
        mega_batch_mult: Optional[int] = None,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        super().__init__(
            dataset=dataset,
            per_device_batch_size=per_device_batch_size,
            length_property=length_property,
            mega_batch_mult=mega_batch_mult,
            seed=seed,
            round_up=round_up,
        )
        assert self.world_size == 1 and self.rank == 0, "Use this class only for single GPU."
        assert per_device_batch_size % 2 == 0, "Batch size must be even: 2k."
        self.k = per_device_batch_size // 2

        assert hasattr(dataset, "idx_affordance") and hasattr(dataset, "idx_chat"), \
            "Dataset must expose idx_affordance and idx_chat."
        self.idx_affordance = list(dataset.idx_affordance)
        self.idx_chat = list(dataset.idx_chat)
        assert len(self.idx_affordance) == len(self.idx_chat), \
            "Dataset must have equal number of affordance and chat samples."

        self.N = len(self.idx_affordance)  # per subset
        # how many balanced batches per epoch
        self.num_batches = math.ceil(self.N / self.k) if self.round_up else (self.N // self.k)
        # override parent's notion of num_samples to match our balanced stream
        self.num_samples = self.num_batches * (2 * self.k)

        # keep the same meaning of mega_batch_mult but per subset:
        # parent group_batch_size ~= mega_batch_mult*(2k)  -> subset ~= mega_batch_mult*k
        self.group_batch_size_subset = max(1, self.group_batch_size // 2) # self.group_batch_size_subset = mega_batch_mult * self.k

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # A: length-group separately inside affordance/chat subsets
        track_lengths = [abs(self.length[i]) for i in self.idx_affordance]
        chat_lengths  = [abs(self.length[i]) for i in self.idx_chat]

        track_pos = get_length_grouped_indices(
            lengths=track_lengths,
            group_batch_size=self.group_batch_size_subset,
            generator=generator,
        )
        chat_pos = get_length_grouped_indices(
            lengths=chat_lengths,
            group_batch_size=self.group_batch_size_subset,
            generator=generator,
        )

        track_order = [self.idx_affordance[p] for p in track_pos]
        chat_order  = [self.idx_chat[p]  for p in chat_pos]

        # make both orders length == num_batches*k (repeat if round_up)
        m = self.num_batches * self.k
        if self.round_up:
            track_order = (track_order * (m // len(track_order) + 1))[:m]
            chat_order  = (chat_order  * (m // len(chat_order)  + 1))[:m]
        else:
            track_order = track_order[:m]
            chat_order  = chat_order[:m]

        # B: interleave into k:k batches, optionally sort within batch by length
        out = []
        for i in range(0, m, self.k):
            batch = track_order[i:i+self.k] + chat_order[i:i+self.k]
            out.extend(batch)

        # keep "biggest first" behavior (like parent)
        if len(out) > 0:
            max_pos = max(range(len(out)), key=lambda t: abs(self.length[out[t]]))
            out[0], out[max_pos] = out[max_pos], out[0]

        assert len(out) == self.num_samples
        return iter(out)
