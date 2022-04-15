# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import numpy as np
from sklearn import preprocessing
from utils import EarlyStopping, EvalMetrics
from sklearn.metrics import recall_score, mean_squared_error, mean_absolute_error

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")


def train(
    train_X,
    optimizer,
    lmse,
    lclass,
    model,
    epoch,
    inputs,
    permutation,
    train_y,
    train_age,
    train_country,
    bs,
):

    classes = [
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

    age_mae, country_uar, y_country, age_ccc, train_val, loss = 0, 0, [], 0, [], 0

    for i in range(0, inputs.size()[0], bs):
        indices = permutation[i : i + bs]
        inputs_X = inputs[indices].to(dev)
        y_pred = model(inputs_X)

        train_pred, logsigma = model(inputs_X)
        lemotion = 0
        for j in classes:
            identifier = classes.index(j)
            targets = torch.from_numpy(
                np.array(train_y.iloc[:, identifier].astype(np.float32))
            )
            targets_y = targets[indices].to(dev)
            loss1 = lmse(train_pred[0][:, identifier].to(dev), targets_y)

            lemotion += loss1

        le = preprocessing.LabelEncoder()
        train_country = le.fit_transform(train_country)
        train_country = torch.from_numpy(np.array(train_country))
        train_age = torch.from_numpy(np.array(train_age))

        lcountry = lclass(train_pred[1].to(dev), train_country[indices].to(dev))
        lage = lmse(
            train_pred[2].type(torch.DoubleTensor).to(dev), train_age[indices].to(dev)
        )

        train_loss = [lemotion, lcountry, lage]
        loss = sum(
            1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2
            for i in range(3)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_val = []
    train_pred, logsigma = model(torch.from_numpy(train_X.astype(np.float32)).to(dev))
    for index, i in enumerate(train_pred):
        train_pred[index] = train_pred[index].cpu()
    for j in classes:
        identifier = classes.index(j)
        ccc_train = EvalMetrics.CCC(
            train_y.iloc[:, identifier],
            train_pred[0][:, identifier].flatten().detach().numpy(),
        )
        train_val.append(ccc_train)
    y_country = torch.max(train_pred[1], 1)
    country_uar = recall_score(
        train_country.cpu(), y_country.indices.cpu(), average="macro"
    )
    age_mae = mean_absolute_error(train_age.cpu(), train_pred[2].detach().numpy())
    age_ccc = EvalMetrics.CCC(train_age.cpu(), train_pred[2].flatten().detach().numpy())

    return age_mae, country_uar, y_country, age_ccc, train_val, loss


def validation(
    lmse,
    lclass,
    model,
    epoch,
    val_inputs,
    val_permutation,
    val_y,
    val_age,
    val_country,
    bs,
):
    classes = [
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
    inputs_X = val_inputs
    y_pred = model(inputs_X)
    val_pred, logsigma = model(inputs_X)

    lemotion = 0
    for j in classes:
        identifier = classes.index(j)
        targets = torch.from_numpy(
            np.array(val_y.iloc[:, identifier].astype(np.float32))
        )
        targets_y = targets.to(dev)
        loss1 = lmse(val_pred[0][:, identifier].to(dev), targets_y)

        lemotion += loss1

    le = preprocessing.LabelEncoder()
    val_country = le.fit_transform(val_country)
    val_country = torch.from_numpy(np.array(val_country))
    val_age = torch.from_numpy(np.array(val_age))

    lcountry = lclass(val_pred[1].to(dev), val_country.to(dev))
    lage = lmse(val_pred[2].type(torch.DoubleTensor).to(dev), val_age.to(dev))

    val_loss = [lemotion, lcountry, lage]
    val_loss = sum(
        1 / (2 * torch.exp(logsigma[i])) * val_loss[i] + logsigma[i] / 2
        for i in range(3)
    )

    return val_loss
