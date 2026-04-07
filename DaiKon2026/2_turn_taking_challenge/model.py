from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils import compute_turn_taking_metrics


class TurnTakingBaseline(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        self.save_hyperparameters({"config": config, "input_dim": input_dim})

        h = config["model"]["hidden_dim"]
        d = config["model"]["dropout"]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(d),
        )
        self.speaker_head = nn.Linear(h, 1)
        self.time_head = nn.Linear(h, 1)

        self.spk_loss = nn.BCEWithLogitsLoss()
        self.time_loss = nn.L1Loss()
        self.reg_w = config["model"]["regression_loss_weight"]

        self.lr = config["training"]["lr"]
        self.weight_decay = config["training"]["weight_decay"]

        self._val_sp = []
        self._val_st = []
        self._val_tp = []
        self._val_tt = []
        self._test_sp = []
        self._test_st = []
        self._test_tp = []
        self._test_tt = []

    def forward(self, x):
        h = self.encoder(x)
        return self.speaker_head(h).squeeze(-1), self.time_head(h).squeeze(-1)

    def _shared_step(self, batch, stage):
        x, y_spk, y_time, _ = batch
        spk_logits, time_pred = self(x)
        loss = self.spk_loss(spk_logits, y_spk) + self.reg_w * self.time_loss(
            time_pred, y_time
        )
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage != "train"),
            on_epoch=True,
            on_step=False,
        )
        return loss, spk_logits, y_spk, time_pred, y_time

    def training_step(self, batch, batch_idx):
        loss, *_ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, spk_logits, y_spk, time_pred, y_time = self._shared_step(batch, "val")
        self._val_sp.append((torch.sigmoid(spk_logits) > 0.5).long().cpu().numpy())
        self._val_st.append(y_spk.long().cpu().numpy())
        self._val_tp.append(time_pred.detach().cpu().numpy())
        self._val_tt.append(y_time.detach().cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_sp:
            return

        metrics = compute_turn_taking_metrics(
            np.concatenate(self._val_sp),
            np.concatenate(self._val_st),
            np.concatenate(self._val_tp),
            np.concatenate(self._val_tt),
        )
        self.log("val_macro_f1", metrics["macro_f1"], prog_bar=True)
        self.log("val_accuracy", metrics["accuracy"], prog_bar=True)
        self.log("val_mae", metrics["mae"], prog_bar=True)

        self._val_sp.clear()
        self._val_st.clear()
        self._val_tp.clear()
        self._val_tt.clear()

    def test_step(self, batch, batch_idx):
        loss, spk_logits, y_spk, time_pred, y_time = self._shared_step(batch, "test")
        self._test_sp.append((torch.sigmoid(spk_logits) > 0.5).long().cpu().numpy())
        self._test_st.append(y_spk.long().cpu().numpy())
        self._test_tp.append(time_pred.detach().cpu().numpy())
        self._test_tt.append(y_time.detach().cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        if not self._test_sp:
            return

        metrics = compute_turn_taking_metrics(
            np.concatenate(self._test_sp),
            np.concatenate(self._test_st),
            np.concatenate(self._test_tp),
            np.concatenate(self._test_tt),
        )
        self.log("test_macro_f1", metrics["macro_f1"])
        self.log("test_accuracy", metrics["accuracy"])
        self.log("test_mae", metrics["mae"])

        self._test_sp.clear()
        self._test_st.clear()
        self._test_tp.clear()
        self._test_tt.clear()

    def configure_optimizers(self):       
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)