from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from dataset import RapportDataModule
from model import RapportBaseline
from utils import ccc_score, mae_score, pearson_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    with open(
        os.path.join(args.run_dir, "config.yaml"),
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    dm = RapportDataModule(config)
    dm.setup()

    if args.split == "test" and dm.test_dataset is None:
        raise ValueError("Test labels are not available in this release. Use --split val.")

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    ckpt = os.path.join(
        ckpt_dir,
        sorted([x for x in os.listdir(ckpt_dir) if x.endswith(".ckpt")])[-1],
    )

    model = RapportBaseline.load_from_checkpoint(
        ckpt,
        config=config,
        input_dim=dm.input_dim,
    ).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds = dm.test_dataset if args.split == "test" else dm.val_dataset
    dl = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()

    preds, targets, meta = [], [], []

    with torch.no_grad():
        for x, y, idx in dl:
            p = model(x.to(device)).cpu().numpy()
            preds.append(p)
            targets.append(y.numpy())

            for i in idx.numpy():
                r = ds.labels.iloc[int(i)]
                meta.append(
                    {
                        "conversation_id": r["conversation_id"],
                        "start_sec": float(r["start_sec"]),
                        "end_sec": float(r["end_sec"]),
                    }
                )

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)

    metrics = {
        "ccc": ccc_score(y_true, y_pred),
        "pearson": pearson_score(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
    }

    pd.DataFrame(meta).assign(
        rapport_pred=y_pred,
        rapport_target=y_true,
    ).to_csv(
        os.path.join(args.run_dir, "predictions", f"{args.split}_predictions.csv"),
        index=False,
    )

    with open(
        os.path.join(args.run_dir, "predictions", f"{args.split}_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()