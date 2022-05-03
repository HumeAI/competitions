import os

import librosa as lb
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class LoadRawData(Dataset):
    
    def __init__(self, 
                 files_path:str,
                 ext:str = 'wav',
                 ):
        """ Creates an instance of the data to use
        
        Args:
          files_path : Path to test files.
          sr         : Sampling rate of the signal.
        """
        self.base_path = files_path
        files_path = list(Path(files_path).glob(f'*.{ext}'))
        self.files_path = [str(x) for x in files_path]
        self.num_input_features = 1600
        
    def __len__(self):
        return len(self.files_path)

    def process_audio(self, raw_wav:np.array):
        """ Split waveform to frames.
        
        Args:
          raw_wav : The raw waveform.
        
        Returns:
          The framed waveform.
        """
        
        # Pad wav to have equal size frames
        pad_size = self.num_input_features - raw_wav.shape[0] % self.num_input_features
        raw_wav = F.pad(raw_wav, (0, pad_size))
        
        # Count number of frames 
        self.num_frames = int(len(raw_wav) / self.num_input_features)
        
        wav_frames = []
        for i in range(1, self.num_frames+1):
            start_frame = (i-1)*self.num_input_features
            end_frame = i*self.num_input_features
            
            wav_frames.append(raw_wav[start_frame:end_frame])
        
        return torch.stack(wav_frames)

    def __getitem__(self, idx:torch.tensor):
        
        data_file = self.files_path[idx]
        
        audio, sr = torchaudio.load(data_file)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        raw_wav = resampler(audio).reshape(-1,)
        raw_wav /= (raw_wav.max() + torch.finfo(torch.float32).eps)
        
        prosessed_audio = self.process_audio(raw_wav)
        
        return prosessed_audio, data_file

