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

from evaluation import Evaluation
from pathlib import Path


torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument("--json_path", type=str, default="./", help="Path to csv files.")
parser.add_argument("--data_path", type=str, default="./", help="Path to wav folder.")
parser.add_argument(
    "--base_dir",
    type=str,
    default="experiments/",
    help="Directory to save model/logs (default experiments).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size to use for the experiment (default 1).",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use GPU (default True)."
)
parser.add_argument(
    "--take_last_frame",
    type=int,
    default=1,
    help="Whether to take the last frame's predictions in the sequence when training the model (default True).",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="experiments/train/model/best_model.pth.tar",
    help="Path to model.",
)


def main(dataset_params, network_params, eval_params):
    evaluation = Evaluation(dataset_params, network_params, eval_params)
    evaluation.start()


def save_arguments(args):
    (Path(args.base_dir) / "eval").mkdir(exist_ok=True, parents=True)

    cmd_args_path = Path(args.base_dir) / "eval" / "cmd_args.txt"

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
        "class": "test_dataset",
        "batch_size": args.batch_size,
        "test": {
            "data_path": args.data_path,
            "json_path": str(Path(args.json_path) / "exvo_test_subject2files.json"),
            "label_names": label_names,
        },
    }

    use_gpu = False if args.use_gpu == 0 else True
    take_last_frame = False if args.take_last_frame == 0 else True
    exp_dir = Path(args.base_dir) / "eval"

    params = {
        "loss_name": "mse",
        "eval_names": ["ccc"],
        "lr": 0.0001,
        "use_gpu": use_gpu,
        "base_dir": str(exp_dir),
        "log_path": f"{str(exp_dir)}/logger.log",
        "model_path": args.model_path,
        "take_last_frame": take_last_frame,
    }

    main(dataset_params, network_params, params)
