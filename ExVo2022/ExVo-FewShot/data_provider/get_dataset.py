# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from pathlib import Path
from .train_dataset import TrainDataset
from .test_dataset import TestDataset


def get_dataset(class_name: str):
    return {"train_dataset": TrainDataset, "test_dataset": TestDataset}[class_name]
