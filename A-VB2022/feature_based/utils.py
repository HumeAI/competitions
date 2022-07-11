# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, mean_squared_error, mean_absolute_error
import torch
from models import MLPReg, MLPClass


class EvalMetrics:
    def CCC(y_true, y_pred):
        x_mean = np.nanmean(y_true, dtype="float32")
        y_mean = np.nanmean(y_pred, dtype="float32")
        x_var = 1.0 / (len(y_true) - 1) * np.nansum((y_true - x_mean) ** 2)
        y_var = 1.0 / (len(y_pred) - 1) * np.nansum((y_pred - y_mean) ** 2)
        cov = np.nanmean((y_true - x_mean) * (y_pred - y_mean))
        return round((2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)

    def MAE(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def MSE(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def UAR(y_true, y_pred):
        return recall_score(y_true, y_pred, average="macro")


class Processing:
    def normalise(scaler, X, y, task):
        train_X = scaler.fit_transform(X[0])
        train_X = pd.DataFrame(train_X).values
        if task != "type":
            train_y, val_y, test_y = (
                y[0].astype(float),
                y[1].astype(float),
                y[2].astype(float),
            )
        else:
            train_y, val_y, test_y = (y[0], y[1], y[2])

        val_X = scaler.transform(X[1])
        test_X = scaler.transform(X[2])
        test_X = pd.DataFrame(test_X).values
        return (
            [train_X, val_X, test_X],
            [train_y, val_y, test_y],
        )


class EarlyStopping:
    # Implementation adapted from
    # https://github.com/Bjarten/early-stopping-pytorch
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0.1,
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class StorePredictions:
    def storehigh(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    ):
        model = MLPReg(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))
        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "Awe": np.array(test_pred[:, 0].detach().numpy()),
            "Excitement": np.array(test_pred[:, 1].detach().numpy()),
            "Amusement": np.array(test_pred[:, 2].detach().numpy()),
            "Awkwardness": np.array(test_pred[:, 3].detach().numpy()),
            "Fear": np.array(test_pred[:, 4].detach().numpy()),
            "Horror": np.array(test_pred[:, 5].detach().numpy()),
            "Distress": np.array(test_pred[:, 6].detach().numpy()),
            "Triumph": np.array(test_pred[:, 7].detach().numpy()),
            "Sadness": np.array(test_pred[:, 8].detach().numpy()),
            "Surprise": np.array(test_pred[:, 9].detach().numpy()),
        }

        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )

    def storetwo(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    ):
        model = MLPReg(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))
        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "Valence": np.array(test_pred[:, 0].detach().numpy()),
            "Arousal": np.array(test_pred[:, 1].detach().numpy()),
        }

        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )

    def storeculture(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    ):
        model = MLPReg(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))

        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "China_Awe": np.array(test_pred[:, 0].detach().numpy()),
            "China_Excitement": np.array(test_pred[:, 1].detach().numpy()),
            "China_Amusement": np.array(test_pred[:, 2].detach().numpy()),
            "China_Awkwardness": np.array(test_pred[:, 3].detach().numpy()),
            "China_Fear": np.array(test_pred[:, 4].detach().numpy()),
            "China_Horror": np.array(test_pred[:, 5].detach().numpy()),
            "China_Distress": np.array(test_pred[:, 6].detach().numpy()),
            "China_Triumph": np.array(test_pred[:, 7].detach().numpy()),
            "China_Sadness": np.array(test_pred[:, 8].detach().numpy()),
            "United States_Awe": np.array(test_pred[:, 9].detach().numpy()),
            "United States_Excitement": np.array(test_pred[:, 10].detach().numpy()),
            "United States_Amusement": np.array(test_pred[:, 11].detach().numpy()),
            "United States_Awkwardness": np.array(test_pred[:, 12].detach().numpy()),
            "United States_Fear": np.array(test_pred[:, 13].detach().numpy()),
            "United States_Horror": np.array(test_pred[:, 14].detach().numpy()),
            "United States_Distress": np.array(test_pred[:, 15].detach().numpy()),
            "United States_Triumph": np.array(test_pred[:, 16].detach().numpy()),
            "United States_Sadness": np.array(test_pred[:, 17].detach().numpy()),
            "South Africa_Awe": np.array(test_pred[:, 18].detach().numpy()),
            "South Africa_Excitement": np.array(test_pred[:, 19].detach().numpy()),
            "South Africa_Amusement": np.array(test_pred[:, 20].detach().numpy()),
            "South Africa_Awkwardness": np.array(test_pred[:, 21].detach().numpy()),
            "South Africa_Fear": np.array(test_pred[:, 22].detach().numpy()),
            "South Africa_Horror": np.array(test_pred[:, 23].detach().numpy()),
            "South Africa_Distress": np.array(test_pred[:, 24].detach().numpy()),
            "South Africa_Triumph": np.array(test_pred[:, 25].detach().numpy()),
            "South Africa_Sadness": np.array(test_pred[:, 26].detach().numpy()),
            "Venezuela_Awe": np.array(test_pred[:, 27].detach().numpy()),
            "Venezuela_Excitement": np.array(test_pred[:, 28].detach().numpy()),
            "Venezuela_Amusement": np.array(test_pred[:, 29].detach().numpy()),
            "Venezuela_Awkwardness": np.array(test_pred[:, 30].detach().numpy()),
            "Venezuela_Fear": np.array(test_pred[:, 31].detach().numpy()),
            "Venezuela_Horror": np.array(test_pred[:, 32].detach().numpy()),
            "Venezuela_Distress": np.array(test_pred[:, 33].detach().numpy()),
            "Venezuela_Triumph": np.array(test_pred[:, 34].detach().numpy()),
            "Venezuela_Sadness": np.array(test_pred[:, 35].detach().numpy()),
            "China_Surprise": np.array(test_pred[:, 36].detach().numpy()),
            "United States_Surprise": np.array(test_pred[:, 37].detach().numpy()),
            "South Africa_Surprise": np.array(test_pred[:, 38].detach().numpy()),
            "Venezuela_Surprise": np.array(test_pred[:, 39].detach().numpy()),
        }

        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )

    def storetype(
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
    ):
        model = MLPClass(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))

        t_pred = torch.max(test_pred, 1)
        t_pred = le.inverse_transform(t_pred.indices.cpu())
        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "Voc_Type": t_pred,
        }
        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )
