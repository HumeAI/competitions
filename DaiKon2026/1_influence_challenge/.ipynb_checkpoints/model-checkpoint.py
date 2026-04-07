from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils import compute_influence_metrics


class InfluenceBaseline(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        self.save_hyperparameters({"config": config, "input_dim": input_dim})

        hidden_dim = config["model"]["hidden_dim"]
        dropout = config["model"]["dropout"]
        num_emotions = config["model"]["num_emotions"]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_emotions)

        self.loss_fn = nn.MSELoss()
        self.lr = config["training"]["lr"]
        self.weight_decay = config["training"]["weight_decay"]

        self._val_preds = []
        self._val_targets = []
        self._test_preds = []
        self._test_targets = []

    def forward(self, x):
        hidden = self.encoder(x)
        logits = self.head(hidden)
        return torch.sigmoid(logits)

    def _shared_step(self, batch, stage):
        x, y, _ = batch

        if not torch.isfinite(x).all():
            raise ValueError("Non-finite values found in input batch x")

        if not torch.isfinite(y).all():
            raise ValueError("Non-finite values found in target batch y")

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage != "train"),
            on_epoch=True,
            on_step=False,
        )

        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._shared_step(batch, "val")

        self._val_preds.append(pred.detach().cpu().numpy())
        self._val_targets.append(y.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        y_pred = np.concatenate(self._val_preds, axis=0)
        y_true = np.concatenate(self._val_targets, axis=0)

        metrics = compute_influence_metrics(y_pred, y_true)

        self.log("val_mean_ccc", metrics["mean_ccc"], prog_bar=True)
        self.log("val_mean_pearson", metrics["mean_pearson"], prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()

    def test_step(self, batch, batch_idx):
        loss, pred, y = self._shared_step(batch, "test")

        self._test_preds.append(pred.detach().cpu().numpy())
        self._test_targets.append(y.detach().cpu().numpy())

        return loss

    def on_test_epoch_end(self):
        if not self._test_preds:
            return

        y_pred = np.concatenate(self._test_preds, axis=0)
        y_true = np.concatenate(self._test_targets, axis=0)

        metrics = compute_influence_metrics(y_pred, y_true)

        self.log("test_mean_ccc", metrics["mean_ccc"])
        self.log("test_mean_pearson", metrics["mean_pearson"])

        self._test_preds.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )