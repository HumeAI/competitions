import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .datasets import LoadRawData


def collate_fn(batch):
    """ Pad batch tensors to have equal length.
    
    Args:
      batch : Data to pad.
    
    Returns:
      padded_seq          : Batched data tensors. 
      num_seqs_per_sample : Number of sequences of each batch tensor.
      data_file           : File name.
    """
    
    data, data_file = zip(*batch)
    
    seqs_to_pad = [torch.Tensor(x) for x in data]
    padded_seqs = pad_sequence(seqs_to_pad, batch_first=True)
    
    num_seqs_per_sample = [len(x) for x in data]
    
    return padded_seqs, num_seqs_per_sample, data_file


def get_dataloader(batch_size:int, shuffle:bool, **kwargs):
    """ Returns the dataloader.
    
    Args:
      batch_size : The batch size to load.
      shuffle    : Whether to shuffle the data.
      **kwargs   : Arguments for the dataset class.
    """
    return DataLoader(LoadRawData(**kwargs),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate_fn)

