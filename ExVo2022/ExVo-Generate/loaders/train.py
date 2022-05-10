"""
Copyright (c) 2022 Marco Jiralerspong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

(MIT License (https://opensource.org/licenses/MIT) )
"""

import os
import torch.utils.data as tdata
import torch
import warnings
import numpy as np
from loaders.datasets_icml import MELDataset, ICMLExVo, PadCollate
from loaders.transforms import RandomTransform, ChangeSpeed, ChangePitch, Shifting, AddNoise

import argparse

if __name__=="__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "", type=str)
    parser.add_argument('--csv_path', default = "", type=str)
    parser.add_argument('--emotion', default = "All", type=str)
    parser.add_argument('--country', default = "All", type=str)

    args = parser.parse_args()

    DATA_DIR = args.data_path
    CSV_DIR = args.csv_path
    BATCH_SIZE = 32
    NUM_EPOCHS = 500

    # Data augmentation
    transf = RandomTransform([ChangeSpeed(), ChangePitch(), Shifting(shift_direction="right"), AddNoise()])
    dataset = ICMLExVo(DATA_DIR, CSV_DIR, emotion=args.emotion, country=args.country, transforms=transf)
    dataset = MELDataset(dataset)

    dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=PadCollate(dim=2, padval=np.log(1e-6)/2.0))

    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            """ TRAIN HERE """
            print(batch)
            break

