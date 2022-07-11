# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from utils import EvalMetrics


def train(
    train_X,
    optimizer,
    lmse,
    lclass,
    model,
    inputs,
    permutation,
    train_y,
    bs,
    classes,
    task,
    dev,
):

    score, loss = [], 0

    for i in tqdm(range(0, inputs.size()[0], bs)):
        indices = permutation[i : i + bs]
        inputs_X = inputs[indices].to(dev)
        train_pred = model(inputs_X.float())
        loss = 0
        if task != "type":
            for j in classes:
                identifier = classes.index(j)
                targets = torch.from_numpy(
                    np.array(train_y.iloc[:, identifier].astype(np.float32))
                )
                targets_y = targets[indices].to(dev)
                loss1 = lmse(train_pred[:, identifier].to(dev), targets_y)
                loss += loss1
        else:
            le = preprocessing.LabelEncoder()
            train_y = le.fit_transform(train_y)
            train_y = torch.from_numpy(np.array(train_y))
            loss = lclass(train_pred.to(dev), train_y[indices].to(dev))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_pred = model(torch.from_numpy(train_X.astype(np.float32)).to(dev))
    for index, i in enumerate(train_pred):
        train_pred[index] = train_pred[index].cpu()
    if task != "type":

        for j in classes:
            identifier = classes.index(j)
            score = EvalMetrics.CCC(
                train_y.iloc[:, identifier],
                train_pred[:, identifier].flatten().detach().numpy(),
            )
    else:
        y_train = torch.max(train_pred, 1)
        score = EvalMetrics.UAR(train_y.cpu(), y_train.indices.cpu())

    return score, loss


def validation(lmse, lclass, model, val_inputs, val_y, classes, task, dev):

    inputs_X = val_inputs
    val_pred = model(inputs_X)

    loss = 0
    if task != "type":

        for j in classes:
            identifier = classes.index(j)
            targets = torch.from_numpy(
                np.array(val_y.iloc[:, identifier].astype(np.float32))
            )
            targets_y = targets.to(dev)
            loss1 = lmse(val_pred[:, identifier].to(dev), targets_y)

            loss += loss1
    else:
        le = preprocessing.LabelEncoder()
        val_y = le.fit_transform(val_y)
        val_y = torch.from_numpy(np.array(val_y))
        loss = lclass(val_pred.to(dev), val_y.to(dev))

    return loss
