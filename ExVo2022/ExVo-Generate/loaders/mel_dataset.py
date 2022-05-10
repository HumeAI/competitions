"""
Copyright (c) 2022 National Film Board of Canada - Office National du Film du Canada

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
   the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
   the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
   promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

(3-Clause BSD License (https://opensource.org/licenses/BSD-3-Clause))
"""

import librosa

import torchaudio
import torch
import numpy as np
import torch.utils.data as tdata
from loaders.transforms import ToTensor


class MELDataset(tdata.Dataset):

    def __init__(self, dataset, stft_hopsize=128, mel_channels=128, sample_rate=16000,
                 transforms=None, pad_length=512, logmag=True, n_samples=None, device="cpu"):

        super(MELDataset, self).__init__()

        self.wav_db = dataset
        self.stft_hopsize = stft_hopsize
        self.mel_channels = mel_channels
        self.sample_rate = sample_rate
        self.n_fft = 4 * mel_channels
        self.n_samples = n_samples
        self.pad_length = pad_length
        self.device = device

        self.logmag = logmag

        # Todo: We can add data augmentation or cleaning techniques here
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate,
                                                            hop_length=stft_hopsize,
                                                            n_fft=self.n_fft,
                                                            n_mels=self.mel_channels)
        # Patch to mel filters to make it invertable with librosa
        self.melspec.mel_scale.fb = torch.tensor(
            librosa.filters.mel(sr=sample_rate, n_mels=self.mel_channels, n_fft=self.n_fft, norm=1).T
        )

        self.transforms = transforms

        self.mels = {}

    def mel2audio(self, mel):
        if self.logmag:
            mel = np.exp(2.0*mel)-1e-6
        return librosa.feature.inverse.mel_to_audio(mel, sr=self.sample_rate, n_fft=self.n_fft,
                                                    hop_length=self.stft_hopsize, norm=1)

    def audio2mel(self, audio):
        mel = self.melspec(audio).detach()
        if self.logmag:
            mel = torch.log(mel+1e-6)/2.0
        return mel

    def __getitem__(self, idx):
        data, sr = self.wav_db[idx]

        # Todo: Parametrize this or resample when not true
        assert sr == self.sample_rate, f'{self.sample_rate} Hz Audio only. Got:{sr}'

        if self.transforms is not None:
            data = self.transforms(data, sr).astype(np.float32)
        data = ToTensor(requires_grad=False)(data)

        mel = self.audio2mel(data)

        # truncate or pad
        if mel.shape[-1]>=self.pad_length:
            mel = mel[:,:,:self.pad_length]
        else:
            mel = pad_tensor(mel, self.pad_length, -1, pad_val=np.log(1e-6)/2.0)
        return mel.detach()
        # return self.mels[idx]

    def __len__(self):
        if self.n_samples:
            return min(self.n_samples, len(self.wav_db))
        return len(self.wav_db)


# Inspired from code here, a dataloader collate function to handle variable length spectrograms
# https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7


def pad_tensor(vec, pad, dim, pad_val=0.0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.ones(*pad_size)*pad_val], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, padval=0.0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.padval = padval

    def pad_tensor(self, vec, pad):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad

        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        dim = self.dim
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.ones(*pad_size)*self.padval], dim=dim)

    def __call__(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max([x.shape[self.dim] for x in batch])
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len) for x in batch]
        # stack all
        x = torch.stack(batch, dim=0)
        x.requires_grad_(False)
        return x