# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)


class TrainDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        data_path: Path,
        label_names: np.array = None,
        sr: int = 16000,
        take_last_frame: bool = False,
        is_training: bool = True,
        window_size: float = 0.1,
    ):
        """Creates an instance of the data to use

        Args:
          csv_path (Path)   : Path to csv file.
          sr (int): Sampling rate of the signal.
        """

        self.label_names = label_names
        self.csv_path = csv_path

        self.wav_folder = Path(data_path)

        self.num_input_features = int(window_size * sr)
        self.num_files = self._get_num_files()

        self.sr = sr

    def _get_num_files(self):
        num_rows = 0
        for row in open(str(self.csv_path)):
            num_rows += 1

        return num_rows

    def frame(
        self, signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1
    ):
        """Equivalent of tf.signal.frame
        code from: https://discuss.pytorch.org/t/pytorch-equivalent-to-tf-signal-frame/123239/2
        """
        signal_length = signal.shape[axis]

        if pad_end:
            frames_overlap = frame_length - frame_step
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(
                frame_length - frames_overlap
            )
            pad_size = int(frame_length - rest_samples)
            if pad_size != 0:
                pad_axis = [0] * signal.ndim
                pad_axis[axis] = pad_size
                signal = F.pad(signal, pad_axis, "constant", pad_value)
        frames = signal.unfold(axis, frame_length, frame_step)

        return frames

    def __len__(self):
        return self.num_files

    def _get_signal(self, idx):
        pd_array = pd.read_csv(
            str(self.csv_path), header=None, skiprows=int(idx), nrows=1
        )
        pd_array = np.array(pd_array).reshape(
            -1,
        )
        data_file = pd_array[0]
        gt = pd_array[1:].astype(np.float32)
        gt = torch.tensor(gt)

        audio_file_path = str(self.wav_folder / data_file)

        # audio: (num_channels x time)
        audio, sr = torchaudio.load(audio_file_path)

        # audio: (time) - assume mono
        audio = audio[0]

        resampler = torchaudio.transforms.Resample(sr, self.sr)
        raw_wav = resampler(audio).reshape(
            -1,
        )

        raw_wav /= torch.abs(raw_wav).max() + torch.finfo(torch.float32).eps

        return (
            raw_wav.view(
                -1,
            ),
            data_file,
            gt,
        )

    def process_audio(self, raw_wav: np.array):
        """Split waveform to frames.

        Args:
          raw_wav : The raw waveform.

        Returns:
          The framed waveform of dims [seq_len, self.num_input_features]
        """

        if len(raw_wav) < 1600:
            # Pad wav to have equal size frames
            pad_size = (
                self.num_input_features - raw_wav.shape[0] % self.num_input_features
            )
            # Dims: [signal_length]
            raw_wav = F.pad(raw_wav, (0, pad_size))

        return self.frame(raw_wav, frame_length=1600, frame_step=1600)

    def __getitem__(self, idx: torch.tensor):

        raw_wav, data_file, gt = self._get_signal(idx)

        # Dim: torch.Size([num_wav_samples]) -> torch.Size([seq_len, self.num_input_features])
        prosessed_audio = self.process_audio(raw_wav)

        gt = gt.unsqueeze(0)
        # Dim: torch.Size([1, 10]) -> torch.Size([seq_len, 10])
        gt = torch.repeat_interleave(
            gt.type(torch.FloatTensor), torch.tensor([prosessed_audio.shape[0]]), dim=0
        )

        seq_len = 200
        if prosessed_audio.shape[0] > seq_len:
            prosessed_audio = prosessed_audio[:seq_len]
            gt = gt[:seq_len]

        return [prosessed_audio], [gt], data_file
