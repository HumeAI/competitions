# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import torch
import torch.nn as nn


class MetricProvider(nn.Module):
    def __init__(self, metric: str = "mse"):
        """Helper class to use metric.

        Args:
            metric (str): Metric to use for evaluation.
        """
        super(MetricProvider, self).__init__()

        self._metric = self._get_metric(metric) if metric else None
        self.eval_fn = self.masked_eval_fn
        self.metric_name = metric

    def _get_metric(self, metric: str):
        """Factory method to return metric.

        Args:
            metric (str): Metric to use for evaluation.
        """

        return {"mae": self.MAE, "mse": self.MSE, "ccc": self.CCC}[metric.lower()]

    def masked_eval_fn(
        self,
        predictions: np.array,
        labels: np.array,
        masks: list,
        take_last_frame: bool,
    ):
        """Method to compute the masked metric evaluation.

        Args:
            predictions (np.array): Model predictions.
            labels (np.array): Data labels.
            masks (list): List of the frames to consider in each batch.
        """

        dtype = predictions[0].dtype

        batch_preds = []
        batch_labs = []

        for i, m in enumerate(masks):
            batch_preds.append(predictions[i][:m])
            batch_labs.append(labels[i][:m])

        return self._metric(batch_preds, batch_labs)

    def create_mask(self, tensor_like, seq_lens, take_last_frame):
        if seq_lens is not None:
            mask = torch.zeros_like(tensor_like, device=tensor_like.device)
            for i, m in enumerate(seq_lens):
                mframe = m - 1 if take_last_frame else range(m)
                mask[i, mframe] = 1.0
        else:
            mask = torch.ones_like(tensor_like, device=tensor_like.device)

        return mask

    def MAE(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask, take_last_frame):

        mask = self.create_mask(y_true, mask, take_last_frame)

        return (mask * torch.abs(y_pred - y_true)).mean()

    def MSE(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mask=None, take_last_frame=1
    ):

        mse = torch.mean((y_pred - y_true) ** 2)

        return mse

    def CCC(
        self,
        y_pred: torch.tensor,
        y_true: torch.tensor,
        seq_lens=None,
        take_last_frame=1,
    ):
        """Concordance Correlation Coefficient (CCC) metric.

        Args:
            predictions (list): Model predictions.
            labels (list): Data labels.
        """

        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)

        y_true_var = torch.var(y_true)
        y_pred_var = torch.var(y_pred)

        mean_cent_prod = torch.mean((y_pred - y_pred_mean) * (y_true - y_true_mean))

        ccc = (2 * mean_cent_prod) / (
            y_pred_var + y_true_var + (y_pred_mean - y_true_mean) ** 2
        )

        return ccc
