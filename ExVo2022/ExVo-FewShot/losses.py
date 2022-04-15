# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn

from functools import partial


class Losses(nn.Module):
    def __init__(self, loss: str = "mse"):
        """Initialize loss object class.

        Args:
          loss (str): Loss function to use
        """
        super(Losses, self).__init__()

        self._loss = self._get_loss(loss) if loss else None
        self.loss_fn = self._loss
        self.loss_name = loss

    def _get_loss(self, loss: str):
        """Factory method to provide the loss function.

        Args:
          loss (str): Name of loss function to use.
        """

        return {"mae": self.mae, "mse": self.mse}[loss.lower()]

    def create_mask(self, tensor_like, take_last_frame, seq_lens=None):
        if seq_lens is not None:
            mask = torch.zeros_like(tensor_like, device=tensor_like.device)

            for i, m in enumerate(seq_lens):
                mframe = m - 1 if take_last_frame else range(m)
                mask[i, mframe] = 1.0
        else:
            mask = torch.ones_like(tensor_like, device=tensor_like.device)

        return mask

    def mae(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        seq_lens=None,
        take_last_frame=True,
    ):

        mask = self.create_mask(y_true, take_last_frame, seq_lens)
        mae = torch.sum(mask * torch.abs(y_pred - y_true)) / torch.sum(mask)

        return mae

    def mse(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        seq_lens=None,
        take_last_frame=True,
    ):

        mask = self.create_mask(y_true, take_last_frame, seq_lens)
        mse = torch.sum(mask * (y_pred - y_true) ** 2) / torch.sum(mask)

        return mse
