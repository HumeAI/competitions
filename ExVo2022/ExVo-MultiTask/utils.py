# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, mean_squared_error, mean_absolute_error


class Processing:
    def normalise(scaler, X, high, age, country):
        train_X = scaler.fit_transform(X[0])
        train_X = pd.DataFrame(train_X).values
        train_y, val_y, test_y = (
            high[0].astype(float),
            high[1].astype(float),
            high[2].astype(float),
        )
        train_country, val_country, test_country = country[0], country[1], country[2]
        train_age, val_age, test_age = (
            age[0].astype(float),
            age[1].astype(float),
            age[2].astype(float),
        )
        val_X = scaler.transform(X[1])
        test_X = scaler.transform(X[2])
        test_X = pd.DataFrame(test_X).values
        return (
            [train_X, val_X, test_X],
            [train_y, val_y, test_y],
            [train_age, val_age, test_age],
            [train_country, val_country, test_country],
        )


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


class EarlyStopping:
    #    Implementation adapted from  https://github.com/Bjarten/early-stopping-pytorch
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
            self.trace_func(f"es: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
