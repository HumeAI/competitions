# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import sys
import argparse
import logging
import torch
import torch.nn as nn
import json

from train import Train
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()

parser.add_argument("--csv_paths", type=str, default="./", help="Path to csv files.")
parser.add_argument("--data_path", type=str, default="./", help="Path to wav files.")
parser.add_argument(
    "--base_dir",
    type=str,
    default="test_experiments/",
    help="Directory to save model/logs (default experiments).",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.0002,
    help="Learning rate to use for the experiment (default 0.0001).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Batch size to use for the experiment (default 8).",
)
parser.add_argument(
    "--number_of_epochs",
    type=int,
    default=50,
    help="Number of epochs to run the experiment (default 100).",
)
parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use GPU (default True)."
)
parser.add_argument(
    "--nepochs2stop",
    type=int,
    default=30,
    help="Number of epochs to stop training if model performance"
    "on validation set has not improved.",
)
parser.add_argument(
    "--take_last_frame",
    type=int,
    default=1,
    help="Whether to take the last frame's predictions in the sequence when training the model (default False).",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Directory to save model/logs (default experiments).",
)


def main(dataset_params, network_params, train_params):
    trainer = Train(dataset_params, network_params, train_params)
    trainer.start()


def save_arguments(args):
    (Path(args.base_dir) / "train").mkdir(exist_ok=True, parents=True)

    cmd_args_path = Path(args.base_dir) / "train" / "cmd_args.txt"
    with open(str(cmd_args_path), "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    args = parser.parse_args()

    label_names = [
        "Awe",
        "Excitement",
        "Amusement",
        "Awkwardness",
        "Fear",
        "Horror",
        "Distress",
        "Triumph",
        "Sadness",
        "Surprise",
    ]

    save_arguments(args)

    network_params = {"hidden_size": 256, "num_layers": 2}

    dataset_params = {
        "class": "train_dataset",
        "batch_size": args.batch_size,
        "train": {
            "data_path": args.data_path,
            "csv_path": str(Path(args.csv_paths) / "exvo_train.csv"),
            "label_names": label_names,
        },
        "valid": {
            "data_path": args.data_path,
            "csv_path": str(Path(args.csv_paths) / "exvo_val.csv"),
            "label_names": label_names,
        },
    }

    use_gpu = False if args.use_gpu == 0 else True
    take_last_frame = False if args.take_last_frame == 0 else True
    exp_dir = Path(args.base_dir) / "train"

    train_params = {
        "number_of_epochs": args.number_of_epochs,
        "lr": args.learning_rate,
        "loss_name": "mse",
        "base_dir": exp_dir,
        "eval_names": ["ccc"],
        "use_gpu": use_gpu,
        "log_path": f"{exp_dir}/logger.log",
        "nepochs2stop": args.nepochs2stop,
        "take_last_frame": take_last_frame,
        "save_summary_steps": 10,
        "model_path": args.model_path,
        "metric2track": "ccc",
    }

    main(dataset_params, network_params, train_params)
