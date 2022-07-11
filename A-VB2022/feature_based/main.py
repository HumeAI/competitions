# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import argparse
import csv
import time
import sys
from pathlib import Path
import random
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch
from torch import nn


from train import validation
from train import train
from dataloader import Dataloader
from utils import EarlyStopping, EvalMetrics, Processing, StorePredictions
from models import MLPReg, MLPClass


warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--directory", help="working directory")
parser.add_argument("-f", "--features", help="feature type")
parser.add_argument("-t", "--task", help="['high','two','culture','type']")
parser.add_argument("-e", "--epochs", default=20, type=int, help="number of epochs")
parser.add_argument(
    "-lr", "--learning_rate", default=0.0001, type=float, help="learning rate"
)
parser.add_argument("-bs", "--batch_size", default=8, type=int, help="batch size")
parser.add_argument(
    "-p", "--patience", default=5, type=int, help="early stopping patience"
)
parser.add_argument(
    "--n_seeds",
    type=int,
    default=5,
    choices=range(1, 6),
    help="number of seeds to try (default: 5, max: 6).",
)
parser.add_argument(
    "--verbose", action="store_true", help="Degree of verbosity, default low"
)
args = parser.parse_args()

feat_dict = {"ComParE": [";", "infer", 2, 6373], "eGeMAPS": [";", "infer", 2, 88]}

feature_type = args.features

label_dict = {
    "high": [
        "high_info",
        [
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
        ],
    ],
    "two": ["two_info", ["Valence", "Arousal"]],
    "culture": [
        "culture_info",
        [
            "China_Awe",
            "China_Excitement",
            "China_Amusement",
            "China_Awkwardness",
            "China_Fear",
            "China_Horror",
            "China_Distress",
            "China_Triumph",
            "China_Sadness",
            "United States_Awe",
            "United States_Excitement",
            "United States_Amusement",
            "United States_Awkwardness",
            "United States_Fear",
            "United States_Horror",
            "United States_Distress",
            "United States_Triumph",
            "United States_Sadness",
            "South Africa_Awe",
            "South Africa_Excitement",
            "South Africa_Amusement",
            "South Africa_Awkwardness",
            "South Africa_Fear",
            "South Africa_Horror",
            "South Africa_Distress",
            "South Africa_Triumph",
            "South Africa_Sadness",
            "Venezuela_Awe",
            "Venezuela_Excitement",
            "Venezuela_Amusement",
            "Venezuela_Awkwardness",
            "Venezuela_Fear",
            "Venezuela_Horror",
            "Venezuela_Distress",
            "Venezuela_Triumph",
            "Venezuela_Sadness",
            "China_Surprise",
            "United States_Surprise",
            "South Africa_Surprise",
            "Venezuela_Surprise",
        ],
    ],
    "type": ["type_info", "Voc_Type"],
}


verbose = args.verbose
task = args.task
data_dir = args.directory
sep_type = feat_dict[feature_type][0]
header_type = feat_dict[feature_type][1]
columns_rm = feat_dict[feature_type][2]
feat_dimensions = feat_dict[feature_type][3]
labels = pd.read_csv(f"labels/{label_dict[task][0]}.csv")
classes = label_dict[task][1]

if "/" in feature_type:
    store_name = feature_type.replace("/", "")
else:
    store_name = feature_type

X, y, val_filename_group, test_filename_group = Dataloader.create(
    store_name, task, data_dir, feature_type, labels, classes, sep_type, columns_rm
)

if task == "type":
    le = preprocessing.LabelEncoder()
    y[0] = le.fit_transform(y[0])
    y[1] = le.transform(y[1])

if verbose:
    print(le.classes_)
scaler = StandardScaler()

X, y = Processing.normalise(scaler, X, y, task)

lr = args.learning_rate
bs = args.batch_size
num_epochs = args.epochs

inputs = torch.from_numpy(X[0].astype(np.float32)).to(dev)
val_inputs = torch.from_numpy(X[1].astype(np.float32)).to(dev)
seed_list = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
seed_list = random.sample(seed_list, args.n_seeds)

score_list = []
timestamp = time.strftime("%d%m%Y-%H%M%S")

val_file_ids, test_file_ids = val_filename_group, test_filename_group

for seed in seed_list:
    if task != "type":
        model = MLPReg(feat_dimensions, len(classes)).to(dev)
    else:
        model = MLPClass(feat_dimensions, len(classes)).to(dev)

    es = EarlyStopping(patience=args.patience, verbose=False, delta=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    torch.manual_seed(seed)

    print(
        f"Task: A-VB {task.capitalize()} | Seed: {seed} | {lr} | {bs} | Max Epochs {num_epochs}"
    )
    lmse, lclass = nn.MSELoss(), nn.CrossEntropyLoss()
    loss_res, val_loss_res, val_result, val_loss = [], [], [], 0

    for epoch in range(num_epochs):
        train_permutation, val_permutation = torch.randperm(
            inputs.size()[0]
        ), torch.randperm(val_inputs.size()[0])

        score, loss = train(
            X[0],
            optimizer,
            lmse,
            lclass,
            model,
            inputs,
            train_permutation,
            y[0],
            bs,
            classes,
            task,
            dev,
        )
        loss_res.append(loss.item())

        val_loss = validation(lmse, lclass, model, val_inputs, y[1], classes, task, dev)

        if task != "type":
            if verbose:
                print(
                    f"{epoch+1}/{num_epochs}\tEmoCCC: {np.round(np.mean(score),3)}\tTrainLoss: {np.round(loss.item(),3)} \tValLoss: {np.round(val_loss.item(),3)}"
                )
        else:
            if verbose:
                print(
                    f"{epoch+1}/{num_epochs}\tUAR: {np.round(np.mean(score),3)}\tTrainLoss: {np.round(loss.item(),3)} \tValLoss: {np.round(val_loss.item(),3)}"
                )

        val_loss_res.append(val_loss.item())
        es(val_loss.item(), model)
        if es.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    val_pred = model(torch.from_numpy(X[1].astype(np.float32)).to(dev))

    for index, i in enumerate(val_pred):
        val_pred[index] = val_pred[index].cpu()

    if task != "type":
        for j in classes:
            identifier = classes.index(j)
            ccc_val = EvalMetrics.CCC(
                y[1].iloc[:, identifier],
                val_pred[:, identifier].flatten().detach().numpy(),
            )
            if verbose:
                print(f"Val CCC \t {j.capitalize()}: \t {ccc_val}", flush=True)
            val_result.append(ccc_val)
        score_list.append(np.mean(val_result))
        print(
            f"------\nEmotion Mean CCC: {np.round(np.mean(val_result),4)}\nSTD: {np.round(np.std(val_result),4)}"
        )
    else:
        pred = torch.max(val_pred, 1)
        uar = EvalMetrics.UAR(y[1], pred.indices.cpu())
        print(f"UAR: {np.round(uar,4)}")
        score_list.append(uar)

        v_pred = le.inverse_transform(pred.indices.cpu())
        val_dict_info = {"File_ID": list(val_file_ids.values), "Voc_Type": v_pred}

        val_prediction_csv = pd.DataFrame.from_dict(val_dict_info).sort_values(
            by="File_ID"
        )
        val_prediction_csv.to_csv(
            f"preds/Validation_A-VB_{timestamp}_{task}_{seed}_{store_name}_lr-{args.learning_rate}_bs-{args.batch_size}.csv",
            index=False,
        )

    torch.save(
        model.state_dict(), f"tmp/{timestamp}_{store_name}_model_{seed}_{task}.pth"
    )

max_score = max(score_list)
index_score = score_list.index(max_score)
seed_score = seed_list[index_score]
print("------")
print(f"Task: A-VB-{task.capitalize()}")
print(f"Max across seeds {np.round(max_score,4)}, seed {seed_score}")
print(f"Mean across seeds {np.round(np.mean(score_list),4)}")
print(f"STD across seeds {np.round(np.std(score_list),4)}")
print("------")

if not Path("preds/").is_file():
    Path("preds/").mkdir(exist_ok=True)

if not Path("results/").is_file():
    Path("results/").mkdir(exist_ok=True)

dict_results = {
    "timestamp": timestamp,
    "set": "validation",
    "feature_type": feature_type,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "max_score": max_score,
    "std_score": np.std(score_list),
    "seed": seed_score,
    "n_seeds": args.n_seeds,
    "task": task,
}
results_csv = pd.DataFrame([dict_results])
results_csv.to_csv(
    f"results/Validation_{timestamp}_{store_name}_{args.learning_rate}_{args.batch_size}_results.csv",
    index=False,
)
print("Test predictions saved")

if task == "high":
    StorePredictions.storehigh(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    )
if task == "two":
    StorePredictions.storetwo(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    )
if task == "culture":
    StorePredictions.storeculture(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    )
if task == "type":
    StorePredictions.storetype(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
        le,
    )
