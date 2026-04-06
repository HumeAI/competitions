"""Evaluate a trained turn-taking model."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from dataset import TurnTakingDataModule
from model import TurnTakingBaseline
from utils import compute_turn_taking_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])
    print(f"Loading checkpoint: {ckpt_path}")

    datamodule = TurnTakingDataModule(config)
    datamodule.setup()
    model = TurnTakingBaseline.load_from_checkpoint(
        ckpt_path,
        config=config,
        input_dim=datamodule.input_dim,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = (
        datamodule.test_dataset if args.split == "test" else datamodule.val_dataset
    )
    loader = (
        datamodule.test_dataloader()
        if args.split == "test"
        else datamodule.val_dataloader()
    )

    all_spk_preds, all_spk_targets = [], []
    all_time_preds, all_time_targets = [], []
    all_conversation_ids, all_prediction_times = [], []

    with torch.no_grad():
        for x, spk_targets, time_targets, indices in loader:
            spk_logits, time_pred = model(x.to(device))
            spk_preds = (torch.sigmoid(spk_logits) > 0.5).long()

            all_spk_preds.append(spk_preds.cpu().numpy())
            all_spk_targets.append(spk_targets.long().numpy())
            all_time_preds.append(time_pred.cpu().numpy())
            all_time_targets.append(time_targets.numpy())

            for i in indices.numpy():
                row = dataset.labels.iloc[int(i)]
                all_conversation_ids.append(row["conversation_id"])
                all_prediction_times.append(float(row["prediction_time"]))

    all_spk_preds = np.concatenate(all_spk_preds)
    all_spk_targets = np.concatenate(all_spk_targets)
    all_time_preds = np.concatenate(all_time_preds)
    all_time_targets = np.concatenate(all_time_targets)

    metrics = compute_turn_taking_metrics(
        all_spk_preds,
        all_spk_targets,
        all_time_preds,
        all_time_targets,
    )

    print(f"{args.split.upper()} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    pred_df = pd.DataFrame(
        {
            "conversation_id": all_conversation_ids,
            "prediction_time": all_prediction_times,
            "speaker_pred": all_spk_preds,
            "speaker_target": all_spk_targets,
            "time_pred": all_time_preds,
            "time_target": all_time_targets,
        }
    )
    pred_path = os.path.join(
        args.run_dir,
        "predictions",
        f"{args.split}_predictions.csv",
    )
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")

    metrics_path = os.path.join(
        args.run_dir,
        "predictions",
        f"{args.split}_metrics.json",
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()