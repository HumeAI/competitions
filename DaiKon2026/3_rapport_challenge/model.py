from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils import ccc_score, pearson_score, mae_score


class RapportBaseline(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        self.save_hyperparameters({"config": config, "input_dim": input_dim})

        hidden_dim = config["model"]["hidden_dim"]
        dropout = config["model"]["dropout"]

        self.lr = float(config["training"]["lr"])
        self.weight_decay = float(config["training"]["weight_decay"])

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.MSELoss()

        self._val_preds = []
        self._val_targets = []
        self._test_preds = []
        self._test_targets = []

    def forward(self, x):
        hidden = self.encoder(x)
        return self.head(hidden).squeeze(-1)

    def _shared_step(self, batch, stage):
        x, y, _ = batch
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

        ccc = ccc_score(y_true, y_pred)
        pearson = pearson_score(y_true, y_pred)
        mae = mae_score(y_true, y_pred)

        self.log("val_ccc", ccc, prog_bar=True)
        self.log("val_pearson", pearson, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)

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

        ccc = ccc_score(y_true, y_pred)
        pearson = pearson_score(y_true, y_pred)
        mae = mae_score(y_true, y_pred)

        self.log("test_ccc", ccc)
        self.log("test_pearson", pearson)
        self.log("test_mae", mae)

        self._test_preds.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )