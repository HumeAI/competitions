"""Training entry point for the Turn-Taking sub-challenge baseline."""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml

from dataset import TurnTakingDataModule
from model import TurnTakingBaseline


def make_run_id(config, seed):
    config_str = f"{config}{seed}{time.time()}"
    short_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{short_hash}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(args.seed)

    run_id = make_run_id(config, args.seed)
    modality = config["data"]["modality"]
    run_dir = os.path.join(config["output"]["run_dir"], modality, run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print(f"Run ID:        {run_id}")
    print(f"Run dir:       {run_dir}")
    print(f"Modality:      {modality}")
    print(f"Data root:     {config['data']['data_root']}")
    print(f"Labels file:   {config['data']['labels_file']}")
    print(f"Context (sec): {config['data']['context_window_s']}")
    print(f"Batch size:    {config['training']['batch_size']}")
    print(f"Max epochs:    {config['training']['max_epochs']}")
    print(f"Device:        {'cuda' if torch.cuda.is_available() else 'cpu'}")

    datamodule = TurnTakingDataModule(config)
    datamodule.setup()

    model = TurnTakingBaseline(config, input_dim=datamodule.input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:    {total_params:,}")

    logger = True
    if args.no_wandb:
        logger = False
    else:
        logger = WandbLogger(
            project="acii-daikon",
            name=f"turn_taking_{modality}_{run_id}",
            config=config,
        )

    callbacks = [
        EarlyStopping(
            monitor="val_macro_f1",
            patience=config["training"]["early_stopping_patience"],
            mode="max",
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch}-{val_macro_f1:.4f}",
            monitor="val_macro_f1",
            mode="max",
            save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    best_path = callbacks[1].best_model_path
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
