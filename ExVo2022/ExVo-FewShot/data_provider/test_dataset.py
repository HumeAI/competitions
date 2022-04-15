# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torch.nn.functional as F
import json

from torch.utils.data import Dataset
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)


class TestDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        data_path: Path,
        n_support: int = 2,
        n_query: int = 1,
        label_names: np.array = None,
        sr: int = 16000,
        window_size: float = 0.1,
    ):

        self.n_query = n_query
        self.n_support = n_support

        self.label_names = label_names
        self.files_path = json.load(open(str(json_path), "r"))

        self.wav_folder = Path(data_path)

        self.num_input_features = int(window_size * sr)
        num_files = len(self.files_path)

        self.len_wav_files = num_files
        self.sr = sr
        self.nfiles = num_files

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
        return self.nfiles

    def _get_subject_files(self, idx):

        subject_selected = sorted(list(self.files_path.keys()))[idx]
        subject_files = self.files_path[subject_selected]

        # Select support/query sets
        subject_files_selected = [subject_files[i] for i in range(self.n_support)]
        subject_files_selected.extend(subject_files[-self.n_query :])
        subject_files_selected = np.array(subject_files_selected)

        # Replace postfix of audio files to 'wav'
        audio_files = subject_files_selected[:, 0]
        gts = subject_files_selected[:, 1:].astype(np.float32)

        audio_file_paths = []
        for x in audio_files:
            audio_file_paths.append(str(self.wav_folder / x))

        return audio_file_paths, torch.tensor(gts)

    def _get_signal(self, idx):
        data_files, ground_truths = self._get_subject_files(idx)

        audio_signals = []
        for x in data_files:
            audio, sr = torchaudio.load(x)
            audio = torch.mean(audio, axis=0)
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            raw_wav = resampler(audio).reshape(
                -1,
            )
            audio_signals.append(raw_wav)

        return audio_signals, data_files, ground_truths

    def process_audio(self, raw_wav: np.array):
        """Split waveform to frames.

        Args:
          raw_wav : The raw waveform.

        Returns:
          The framed waveform of dims [seq_len, self.num_input_features]
        """

        # Pad wav to have equal size frames
        pad_size = self.num_input_features - raw_wav.shape[0] % self.num_input_features

        # Dims: [signal_length]
        raw_wav = F.pad(raw_wav, (0, pad_size))

        return self.frame(raw_wav, frame_length=1600, frame_step=1600)

    def __getitem__(self, idx: torch.tensor):

        audio_signals, data_file, gts = self._get_signal(idx)

        # Dim: torch.Size([num_wav_samples]) -> torch.Size([seq_len, self.num_input_features])
        prosessed_audios = []
        rep_ground_truth = []
        for x, gt in zip(*[audio_signals, gts]):
            prosessed_audio = self.process_audio(x)

            # Dim: torch.Size([1, 48]) -> torch.Size([seq_len, 48])
            gt = torch.repeat_interleave(
                gt.type(torch.FloatTensor).view(1, -1),
                torch.tensor([prosessed_audio.shape[0]]),
                dim=0,
            )

            prosessed_audios.append(prosessed_audio)
            rep_ground_truth.append(gt)

        return prosessed_audios, rep_ground_truth, data_file
