# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch import nn


class MLPClass(nn.Module):
    def __init__(self, feat_dimensions, output_len):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dimensions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
        )

        self.class_output = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, output_len)
        )

    def forward(self, x):
        main_mlp = self.mlp(x)
        output = self.class_output(main_mlp)
        return output


class MLPReg(nn.Module):
    def __init__(self, feat_dimensions, output_len):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dimensions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
        )

        self.high_output = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, output_len)
        )

    def forward(self, x):
        main_mlp = self.mlp(x)
        output = torch.sigmoid(self.high_output(main_mlp))
        return output
