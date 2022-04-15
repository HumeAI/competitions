# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .get_dataset import get_dataset
from itertools import chain


def collate_fn(batch):
    """Pad batch tensors to have equal length.

    Args:
      batch (list): Data to pad.

    Returns:
      padded_seq (torch.Tensor): Batched data tensors.
      gt (torch.Tensor): Ground truth values.
      num_seqs_per_sample (list): Number of sequences of each batch tensor.
      data_file (str): File names.
    """

    data, ground_truth, filenames = zip(*batch)

    seqs_to_pad, gt = [], []
    num_seqs_per_sample = []
    for i, subject_wavs in enumerate(data):
        for j, audio in enumerate(subject_wavs):
            seqs_to_pad.append(audio)
            num_seqs_per_sample.append(len(audio))
            gt.append(ground_truth[i][j])

    padded_seqs = pad_sequence(seqs_to_pad, batch_first=True)
    gt = pad_sequence(gt, batch_first=True)

    return padded_seqs, gt, num_seqs_per_sample, list(chain.from_iterable(filenames))


def get_dataloader(dataset_class: str, batch_size: int, shuffle: bool, **kwargs):
    """Returns the dataloader.

    Args:
      dataset_name (str): Name of the dataset to use.
      batch_size (int): The batch size to load.
      shuffle (bool): Whether to shuffle the data.
    """
    DatasetClass = get_dataset(dataset_class)

    return DataLoader(
        DatasetClass(**kwargs),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
