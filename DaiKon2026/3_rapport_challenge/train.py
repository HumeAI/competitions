from __future__ import annotations

import argparse
import hashlib
import os
import time

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import RapportDataModule
from model import RapportBaseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(args.seed)

    run_id = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{hashlib.md5(f'{config}{args.seed}{time.time()}'.encode()).hexdigest()[:8]}"
    )
    modality = config["data"]["modality"]
    run_dir = os.path.join(config["output"]["run_dir"], modality, run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    dm = RapportDataModule(config)
    dm.setup()
    model = RapportBaseline(config, input_dim=dm.input_dim)

    logger = (
        False
        if args.no_wandb
        else WandbLogger(
            project="acii-daikon",
            name=f"rapport_{modality}_{run_id}",
            config=config,
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=logger,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(
                monitor="val_ccc",
                patience=config["training"]["early_stopping_patience"],
                mode="max",
            ),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val_ccc",
                mode="max",
                save_top_k=1,
            ),
        ],
    )

    trainer.fit(model, datamodule=dm)

    if dm.test_dataset is not None:
        trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        print("No test labels available; skipping trainer.test().")


if __name__ == "__main__":
    main()