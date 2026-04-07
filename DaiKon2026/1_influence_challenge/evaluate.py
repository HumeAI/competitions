from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from dataset import InfluenceDataModule
from model import InfluenceBaseline
from utils import EMOTIONS, compute_influence_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dm = InfluenceDataModule(config)
    dm.setup()

    if args.split == "test" and dm.test_dataset is None:
        raise ValueError("Test labels are not available in this release. Use --split val.")

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    ckpt_files = sorted(
        [filename for filename in os.listdir(ckpt_dir) if filename.endswith(".ckpt")]
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])

    model = InfluenceBaseline.load_from_checkpoint(
        ckpt_path,
        config=config,
        input_dim=dm.input_dim,
    ).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.split == "test":
        dataset = dm.test_dataset
        dataloader = dm.test_dataloader()
    else:
        dataset = dm.val_dataset
        dataloader = dm.val_dataloader()

    preds = []
    targets = []
    meta = []

    with torch.no_grad():
        for x, y, idx in dataloader:
            pred = model(x.to(device)).cpu().numpy()

            preds.append(pred)
            targets.append(y.numpy())

            for i in idx.numpy():
                row = dataset.labels.iloc[int(i)]
                meta.append(
                    {
                        "conversation_id": row["conversation_id"],
                        "speaker_id": row["speaker_id"],
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                    }
                )

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)

    metrics = compute_influence_metrics(y_pred, y_true)

    pred_df = pd.DataFrame(meta)
    for i, emotion in enumerate(EMOTIONS):
        pred_df[f"{emotion}_pred"] = y_pred[:, i]
        pred_df[f"{emotion}_target"] = y_true[:, i]

    predictions_dir = os.path.join(args.run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    pred_csv = os.path.join(predictions_dir, f"{args.split}_predictions.csv")
    metrics_json = os.path.join(predictions_dir, f"{args.split}_metrics.json")

    pred_df.to_csv(pred_csv, index=False)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions to: {pred_csv}")
    print(f"Saved metrics to: {metrics_json}")


if __name__ == "__main__":
    main()