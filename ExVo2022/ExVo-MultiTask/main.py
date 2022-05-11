# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import argparse
import warnings
import os
import csv
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import jsonlines

from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from scipy.stats import hmean

import torch
import torch.optim as optim
import torch.nn as nn

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    
from dataloader import Dataloader
from utils import Processing, EvalMetrics, EarlyStopping
from models import MultiTask
from train import train
from train import validation
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--directory", help="working directory")
parser.add_argument("-l", "--labels", help="label file, data_info.csv")
parser.add_argument("-f", "--features", help="feature type")
parser.add_argument("-e", "--epochs", default=20, type=int, help="No of Epochs, Baseline 20")
parser.add_argument("-lr", "--learningrate", default=0.001, type=float, help="Learning rate, Baseline 0.001")
parser.add_argument("-bs", "--batchsize", default=8, type=int, help="Batch Size, Baseline 8")
parser.add_argument("-p", "--patience", default=5, type=int, help="Early stopping patience, Baseline 5")
parser.add_argument("-tn", "--teamname", type=str, default='Baseline', help="Name of team")
parser.add_argument("--store_pred", action="store_true", help="Store test set predictions")
parser.add_argument("--save_csv", action="store_true", help="store overview of results")
parser.add_argument("--pltloss", action="store_true", help="plot training loss")
parser.add_argument("--n_seeds", type=int, default=3,choices=range(1, 6),help="number of seeds to try (default: 3, max: 6).")
parser.add_argument("-ckpt", "--checkpoint_fname", type=str, default='', help="Checkpoint for validation")
args = parser.parse_args()


def baseline(
    plot_loss,
    timestamp,
    classes,
    feature_type,
    X,
    emo_y,
    age_y,
    country_y,
    feat_dimensions,
    num_epochs,
    lr,
    bs,
    es_patience,
    seed,
    store_name,
):
    print(f"Running experiments with {feature_type}")
    lmse, lclass = nn.MSELoss(), nn.CrossEntropyLoss()
    es_delta = 0.1
    val_result, loss_res, val_loss_res = [], [], []
    
    model = MultiTask(feat_dimensions).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    inputs = torch.from_numpy(X[0].astype(np.float32)).to(dev)
    val_inputs = torch.from_numpy(X[1].astype(np.float32)).to(dev)

    torch.manual_seed(seed)

    print(f"Seed {seed} | {lr} | Batch Size {bs} | Epochs {num_epochs}")

    es = EarlyStopping(patience=es_patience, verbose=False, delta=es_delta)
    for epoch in range(num_epochs):
        train_permutation = torch.randperm(inputs.size()[0]) 
        val_permutation = torch.randperm(val_inputs.size()[0])

        age_mae, country_uar, y_country, age_ccc, train_val, loss = train(
            X[0],
            optimizer,
            lmse,
            lclass,
            model,
            epoch,
            inputs,
            train_permutation,
            emo_y[0],
            age_y[0],
            country_y[0],
            bs,
        )
        loss_res.append(loss.item())

        val_loss = validation(
            lmse,
            lclass,
            model,
            epoch,
            val_inputs,
            val_permutation,
            emo_y[1],
            age_y[1],
            country_y[1],
            bs,
        )
        print(
            f"{epoch+1}/{num_epochs}\tEmoCCC: {np.round(np.mean(train_val),3)}\tCountryUAR: {np.round(country_uar,3)}\t AgeMAE: {np.round(age_mae,3)}\tTrainLoss: {np.round(loss.item(),3)} \tValLoss: {np.round(val_loss.item(),3)}"
        )
        val_loss_res.append(val_loss.item())

        es(val_loss.item(), model)
        if es.early_stop:
            print(f"Early stopping {epoch}")
            break

    val_pred, logsigma = model(torch.from_numpy(X[1].astype(np.float32)).to(dev))

    for index, i in enumerate(val_pred):
        val_pred[index] = val_pred[index].cpu()
    for j in classes:
        identifier = classes.index(j)
        ccc_val = EvalMetrics.CCC(
            emo_y[1].iloc[:, identifier],
            val_pred[0][:, identifier].flatten().detach().numpy(),
        )
        print(f"Val CCC \t {j.capitalize()}: \t {ccc_val}", flush=True)
        val_result.append(ccc_val)
    print(
        f"------\nEmotion Mean CCC: {np.round(np.mean(val_result),4)}\nSTD: {np.round(np.std(val_result),4)}"
    )

    le = preprocessing.LabelEncoder()
    val_country = le.fit_transform(country_y[1])
    y_country = torch.max(val_pred[1], 1)
    country_uar = EvalMetrics.UAR(country_y[1], y_country.indices)
    print(f"Country UAR: {np.round(country_uar,4)}")

    age_mae = EvalMetrics.MAE(age_y[1], val_pred[2].detach().numpy())
    inverted_mae = 1 / age_mae
    print(f"Age MAE: {np.round(age_mae,4)}\n~MAE: {np.round(inverted_mae,4)}\n------")
    try:
        val_hmean_score = hmean([np.mean(val_result), country_uar, inverted_mae])
        print(f"HMean: {np.round(val_hmean_score,4)}\n------")
    except:
        print("HMean not possible")

    if plot_loss:
        sns.lineplot(data=loss_res)
        sns.lineplot(data=val_loss_res)
        plt.legend(["Train", "Val"])
        plt.title(f"Loss, No Epochs {epoch+1}")
        plt.xlabel("Epoch No.")
        plt.ylabel("Combined Loss")
        plt.savefig(f"plots/{timestamp}_{num_epochs}_{lr}_{bs}_{store_name}_{seed}.png")
        plt.clf()
    torch.save(model.state_dict(), f"tmp/{timestamp}_{store_name}_model_{seed}.pth")

    return (
        model,
        np.round(val_hmean_score, 4),
        [np.mean(val_result), country_uar, inverted_mae],
    )


def store_predictions(
    feat_dimensions,
    X,
    team_name,
    labels,
    classes,
    seed,
    timestamp,
    feature_type,
    country_y,
    test_filename_group,
    store_name,
):
    print(f"Predicting on the test set...")
    submission_no = 1  # change manually

    model = MultiTask(feat_dimensions).to(dev)
    model.load_state_dict(torch.load(f"tmp/{timestamp}_{store_name}_model_{seed}.pth"))

    file_ids = test_filename_group
    test_pred, logsigma = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))

    for index, i in enumerate(test_pred):
        test_pred[index] = test_pred[index].cpu()
    le = preprocessing.LabelEncoder()

    test_country = le.fit_transform(country_y[2])
    y_country_pred = torch.max(test_pred[1], 1)

    dict_info = {
        "File_ID": list(file_ids.values),
        "Country": y_country_pred[1],
        "Age": test_pred[2].flatten().detach().numpy(),
        "Awe": np.array(test_pred[0][:, 0].detach().numpy()),
        "Excitement": np.array(test_pred[0][:, 1].detach().numpy()),
        "Amusement": np.array(test_pred[0][:, 2].detach().numpy()),
        "Awkwardness": np.array(test_pred[0][:, 3].detach().numpy()),
        "Fear": np.array(test_pred[0][:, 4].detach().numpy()),
        "Horror": np.array(test_pred[0][:, 5].detach().numpy()),
        "Distress": np.array(test_pred[0][:, 6].detach().numpy()),
        "Triumph": np.array(test_pred[0][:, 7].detach().numpy()),
        "Sadness": np.array(test_pred[0][:, 8].detach().numpy()),
        "Surprise": np.array(test_pred[0][:, 9].detach().numpy()),
    }

    prediction_csv = pd.DataFrame.from_dict(dict_info)

    prediction_csv.to_csv(
        f"preds/ExVo-Multi_{team_name}_{str(submission_no)}_{seed}_{store_name}.csv",
        index=False,
    )


def store_val_predictions(
    feat_dimensions,
    X,
    labels,
    classes,
    feature_type,
    emo_y,
    age_y,
    country_y,
    test_filename_group,
    checkpoint_fname
):
    print(f"Predicting on the validation set...")

    model = MultiTask(feat_dimensions).to(dev)
    model.eval()
    model.load_state_dict(torch.load(f"tmp/{checkpoint_fname}"))

    file_ids = test_filename_group
    print(file_ids)
    
    test_pred, logsigma = model(torch.from_numpy(X[1].astype(np.float32)).to(dev))

    for index, i in enumerate(test_pred):
        test_pred[index] = test_pred[index].cpu()
    le = preprocessing.LabelEncoder()

    test_country = le.fit_transform(country_y[2])
    y_country_pred = torch.max(test_pred[1], 1)

    # label_df = labels[labels['Split'] == 'Val'].astype(dtype={'Country': 'int', 'Age': 'float',
    #                       'Amusement': 'float', 'Awe': 'float',
    #                       'Awkwardness': 'float', 'Distress': 'float',
    #                       'Excitement': 'float', 'Fear': 'float',
    #                       'Horror': 'float', 'Sadness': 'float',
    #                       'Surprise': 'float', 'Triumph': 'float'})
    #
    # label_df = label_df.rename(columns={'File_ID': 'id', 'Subject_ID': 'speaker_id',
    #                         'Age': 'age', 'Country': 'country',
    #                         'Country_string': 'country_str'})
    #
    # label_df['id'] = label_df['id'].apply(lambda x: x.replace("[", "").replace("]", "") + '.wav')
    # label_df['audio'] = label_df['id']
    # label_df['audio_path'] = label_df['id'].apply(lambda x: '/data2/atom/datasets/exvo/wav/' + x)
    # label_df['speaker_id'] = label_df['speaker_id'].apply(lambda x: int(x.split('Speaker_')[-1]) if isinstance(x, str) else x)
    # label_df['country_str'] = label_df['country_str'].apply(lambda x: x.strip().lower() if isinstance(x, str) else str(x))
    # label_df['emotion_intensity'] = label_df[['Amusement', 'Awe',
    #                               'Awkwardness', 'Distress',
    #                               'Excitement', 'Fear',
    #                               'Horror', 'Sadness',
    #                               'Surprise', 'Triumph']].values.tolist()
    #
    # label_df = label_df.drop(['Amusement', 'Awe',
    #               'Awkwardness', 'Distress',
    #               'Excitement', 'Fear',
    #               'Horror', 'Sadness',
    #               'Surprise', 'Triumph'], axis=1)
    #
    # print(label_df.columns)

    print(country_y[1])

    label_dict_info = {
        'id': file_ids['0'].tolist(),
        "country": country_y[1]['0'].astype(int).tolist(),
        "age": age_y[1]['0'].tolist(),
        "Awe": emo_y[1].iloc[:, 0].tolist(),
        "Excitement": emo_y[1].iloc[:, 1].tolist(),
        "Amusement": emo_y[1].iloc[:, 2].tolist(),
        "Awkwardness": emo_y[1].iloc[:, 3].tolist(),
        "Fear": emo_y[1].iloc[:, 4].tolist(),
        "Horror": emo_y[1].iloc[:, 5].tolist(),
        "Distress": emo_y[1].iloc[:, 6].tolist(),
        "Triumph": emo_y[1].iloc[:, 7].tolist(),
        "Sadness": emo_y[1].iloc[:, 8].tolist(),
        "Surprise": emo_y[1].iloc[:, 9].tolist(),
    }

    label_df = pd.DataFrame.from_dict(label_dict_info)

    label_df['id'] = label_df['id'].apply(lambda x: str(x) + '.wav')
    label_df['audio'] = label_df['id']
    label_df['audio_path'] = label_df['id'].apply(lambda x: '/data2/atom/datasets/exvo/wav/' + x)
    label_df['emotion_intensity'] = label_df[['Amusement', 'Awe',
                                  'Awkwardness', 'Distress',
                                  'Excitement', 'Fear',
                                  'Horror', 'Sadness',
                                  'Surprise', 'Triumph']].values.tolist()

    label_df = label_df.drop(['Amusement', 'Awe',
                  'Awkwardness', 'Distress',
                  'Excitement', 'Fear',
                  'Horror', 'Sadness',
                  'Surprise', 'Triumph'], axis=1)
    
    print(label_df)

    print(label_df.columns)

    dict_info = {
        # "File_ID": list(file_ids.values),
        "country_pred_label": y_country_pred[1],
        "age_pred_label": test_pred[2].flatten().detach().numpy(),
        "Awe": np.array(test_pred[0][:, 0].detach().numpy()),
        "Excitement": np.array(test_pred[0][:, 1].detach().numpy()),
        "Amusement": np.array(test_pred[0][:, 2].detach().numpy()),
        "Awkwardness": np.array(test_pred[0][:, 3].detach().numpy()),
        "Fear": np.array(test_pred[0][:, 4].detach().numpy()),
        "Horror": np.array(test_pred[0][:, 5].detach().numpy()),
        "Distress": np.array(test_pred[0][:, 6].detach().numpy()),
        "Triumph": np.array(test_pred[0][:, 7].detach().numpy()),
        "Sadness": np.array(test_pred[0][:, 8].detach().numpy()),
        "Surprise": np.array(test_pred[0][:, 9].detach().numpy()),
    }

    prediction_df = pd.DataFrame.from_dict(dict_info)

    prediction_df['emotion_intensity_pred_label'] = prediction_df[['Amusement', 'Awe',
                                                                   'Awkwardness', 'Distress',
                                                                   'Excitement', 'Fear',
                                                                   'Horror', 'Sadness',
                                                                   'Surprise', 'Triumph']].values.tolist()

    prediction_df = prediction_df.drop(['Amusement', 'Awe',
                                        'Awkwardness', 'Distress',
                                        'Excitement', 'Fear',
                                        'Horror', 'Sadness',
                                        'Surprise', 'Triumph'], axis=1)

    prediction_df['input'] = pd.Series(label_df.to_dict(orient='records'))

    prediction_fname = checkpoint_fname.split('.')[0]

    # prediction_df.to_csv(
    #     f"preds/ExVo-Multi_val_{prediction_fname}.csv",
    #     index=False,
    # )

    write_jsonl_into_file(prediction_df.to_dict(orient='records'), f"preds/ExVo-Multi_val_{prediction_fname}.jsonl")

    val_pred = test_pred
    val_result = []

    for j in classes:
        identifier = classes.index(j)
        ccc_val = EvalMetrics.CCC(
            emo_y[1].iloc[:, identifier],
            val_pred[0][:, identifier].flatten().detach().numpy(),
        )
        print(f"Val CCC \t {j.capitalize()}: \t {ccc_val}", flush=True)
        val_result.append(ccc_val)
    print(
        f"------\nEmotion Mean CCC: {np.round(np.mean(val_result),4)}\nSTD: {np.round(np.std(val_result),4)}"
    )

    le = preprocessing.LabelEncoder()
    val_country = le.fit_transform(country_y[1])
    y_country = torch.max(val_pred[1], 1)
    country_uar = EvalMetrics.UAR(country_y[1], y_country.indices)
    print(f"Country UAR: {np.round(country_uar,4)}")

    age_mae = EvalMetrics.MAE(age_y[1], val_pred[2].detach().numpy())
    
    inverted_mae = 1 / age_mae
    print(f"Age MAE: {np.round(age_mae,4)}\n~MAE: {np.round(inverted_mae,4)}\n------")


def write_jsonl_into_file(data, fname):
    with jsonlines.open(str(fname), mode='w') as f:
        for line in data:
            f.write(line)


def main():
    data_dir = args.directory
    labels = pd.read_csv(f"{data_dir}{args.labels}")
    feature_type = args.features
    plot_loss = args.pltloss
    timestamp = time.strftime("%d%m%Y-%H%M%S")
    targets = [
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
        "Age",
        "Country",
    ]
    classes = targets[:10]

    plot_folder = Path("plots/")
    if not plot_folder.is_file():
        plot_folder.mkdir(exist_ok=True)

    pred_folder = Path("preds/")
    if not pred_folder.is_file():
        pred_folder.mkdir(exist_ok=True)

    results_folder = Path("results/")
    if not results_folder.is_file():
        results_folder.mkdir(exist_ok=True)

    # DataLoader
    if "/" in feature_type:
        store_name = feature_type.replace("/", "")
    else:
        store_name = feature_type

    if not os.path.exists(f"tmp/{store_name}_train_X.csv"):
        X, high, age, country, feat_dimensions, test_filename_group = Dataloader.create(
            labels,
            True,
            data_dir,
            feature_type,
            targets[:10],
            targets[10],
            targets[11],
            store_name,
        )
    else:
        X, high, age, country, feat_dimensions, test_filename_group = Dataloader.load(
            feature_type, store_name
        )

    # Scale Data
    scaler = StandardScaler()

    X, emo_y, age_y, country_y = Processing.normalise(scaler, X, high, age, country)

    hmean_list, ccc_list, uar_list, mae_list = [], [], [], []

    seed_list = [101,102,103,104,105,106]
    seed_list = random.sample(seed_list, args.n_seeds)
    for seed in seed_list:
        print(f"Running Model for {len(seed_list)} seeds")
        model, hmean, metrics = baseline(
            plot_loss,
            timestamp,
            classes,
            feature_type,
            X,
            emo_y,
            age_y,
            country_y,
            feat_dimensions,
            args.epochs,
            args.learningrate,
            args.batchsize,
            args.patience,
            seed,
            store_name,
        )
        if args.store_pred:
            store_predictions(
                feat_dimensions,
                X,
                args.teamname,
                labels,
                classes,
                seed,
                timestamp,
                feature_type,
                country_y,
                test_filename_group,
                store_name,
            )

        hmean_list.append(hmean)
        ccc_list.append(metrics[0])
        uar_list.append(metrics[1])
        mae_list.append(metrics[2])

    max_hmean = np.max(hmean_list)
    max_index_hmean = np.argmax(hmean_list)
    seed_best, std_hmean = seed_list[max_index_hmean], np.std(hmean_list)
    ccc_best, uar_best, mae_best = (
        ccc_list[max_index_hmean],
        uar_list[max_index_hmean],
        mae_list[max_index_hmean],
    )

    print(
        f"Harmonic Mean, Validation Best with seed [{seed_best}]: {max_hmean}, STD: {np.round(std_hmean,4)}"
    )
    print(
        f"For this Harmonic Mean, CCC {np.round(ccc_best,4)} | UAR {np.round(uar_best,4)} | MAE {np.round(mae_best,4)}"
    )

    if args.save_csv:
        dict_results = {
            "Timestamp": timestamp,
            "Feature Type": feature_type,
            "Learning Rate": args.learningrate,
            "Batch Size": args.batchsize,
            "Max HMean": max_hmean,
            "Std HMean": std_hmean,
            "Seed": seed_best,
            "n_seeds": args.n_seeds,
            f"CCC_{seed_best}": ccc_best,
            f"UAR_{seed_best}": uar_best,
            f"MAE_{seed_best}": mae_best,
        }
        results_csv = pd.DataFrame([dict_results])
        results_csv.to_csv(
            f"results/{timestamp}_{store_name}_{args.learningrate}_{args.batchsize}_results.csv",
            index=False,
        )


def val_main():
    data_dir = args.directory
    labels = pd.read_csv(f"{data_dir}{args.labels}")
    feature_type = args.features
    plot_loss = args.pltloss
    timestamp = time.strftime("%d%m%Y-%H%M%S")
    targets = [
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
        "Age",
        "Country",
    ]
    classes = targets[:10]

    plot_folder = Path("plots/")
    if not plot_folder.is_file():
        plot_folder.mkdir(exist_ok=True)

    pred_folder = Path("preds/")
    if not pred_folder.is_file():
        pred_folder.mkdir(exist_ok=True)

    results_folder = Path("results/")
    if not results_folder.is_file():
        results_folder.mkdir(exist_ok=True)

    # DataLoader
    if "/" in feature_type:
        store_name = feature_type.replace("/", "")
    else:
        store_name = feature_type

    if not os.path.exists(f"tmp/{store_name}_val_filename.csv"):
        X, high, age, country, feat_dimensions, test_filename_group, val_filename_group = Dataloader.create(
            labels,
            True,
            data_dir,
            feature_type,
            targets[:10],
            targets[10],
            targets[11],
            store_name,
            return_val=True
        )
    else:
        X, high, age, country, feat_dimensions, test_filename_group, val_filename_group = Dataloader.load(
            feature_type, store_name, return_val=True
        )

    # Scale Data
    scaler = StandardScaler()

    X, emo_y, age_y, country_y = Processing.normalise(scaler, X, high, age, country)

    store_val_predictions(
        feat_dimensions,
        X,
        labels,
        classes,
        feature_type,
        emo_y,
        age_y,
        country_y,
        val_filename_group,
        args.checkpoint_fname,
    )


if __name__ == "__main__":
    if args.checkpoint_fname == '':
        main()
    else:
        val_main()
