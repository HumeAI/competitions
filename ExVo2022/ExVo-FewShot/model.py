# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size=256, num_layers=2):
        super(Model, self).__init__()
        self.cnn_block1 = self.cnn_block(
            in_channels=1, out_channels=64, kernel_size=8, stride=1, padding=3
        )
        self.mp1 = nn.MaxPool1d(kernel_size=10, stride=10)

        self.cnn_block2 = self.cnn_block(
            in_channels=64, out_channels=128, kernel_size=6, stride=1, padding=2
        )
        self.mp2 = nn.MaxPool1d(kernel_size=8, stride=8)

        self.cnn_block3 = self.cnn_block(
            in_channels=128, out_channels=256, kernel_size=6, stride=1, padding=2
        )
        self.mp3 = nn.MaxPool1d(kernel_size=8, stride=8)

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, 10)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters of the model."""
        for m in list(self.modules()):
            self._init_weights(m)

    def _init_weights(self, m):
        """Helper method to initialize the parameters of the model
        with Kaiming uniform initialization.
        """

        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.kaiming_uniform_(param)

    def cnn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        batch_size, seq_length, t = x.shape
        x = x.view(batch_size * seq_length, 1, t)

        cnn_out1 = self.cnn_block1(x)
        mp_out1 = self.mp1(cnn_out1)

        cnn_out2 = self.cnn_block2(mp_out1)
        mp_out2 = self.mp2(cnn_out2)

        cnn_out3 = self.cnn_block3(mp_out2)
        mp_out3 = self.mp3(cnn_out3)
        audio_out = mp_out3.view(batch_size, seq_length, -1)

        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(audio_out)

        predictions = self.linear(rnn_out)

        return predictions
