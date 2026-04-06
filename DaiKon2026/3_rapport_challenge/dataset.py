from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


@lru_cache(maxsize=256)
def _load_audio(path: str):
    df = pq.read_table(path).to_pandas()
    if len(df) == 0:
        return None, None
    return (
        df["timestamp"].to_numpy(dtype=np.float32),
        np.stack(df["embedding"].values).astype(np.float32),
    )


@lru_cache(maxsize=512)
def _load_video(path: str):
    df = pq.read_table(path).to_pandas()
    if len(df) == 0:
        return None, None
    return (
        df["timestamp"].to_numpy(dtype=np.float32),
        np.stack(df["embedding"].values).astype(np.float32),
    )


def _pool(ts, emb, start_t, end_t, fallback_dim=None):
    if ts is None or emb is None:
        return np.zeros((fallback_dim,), dtype=np.float32)

    idx = np.where((ts >= start_t) & (ts <= end_t))[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(ts - 0.5 * (start_t + end_t))))])

    return np.nan_to_num(emb[idx].mean(axis=0).astype(np.float32))


class RapportDataset(Dataset):
    def __init__(self, labels_df, data_root, modality="audio"):
        self.labels = labels_df.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.modality = modality

        keep = []
        for i, row in self.labels.iterrows():
            conv = self.data_root / row["conversation_id"]
            ok = True

            if modality in ["audio", "multimodal"]:
                ok = ok and (
                    conv / "features" / "audio" / "whisper_small.parquet"
                ).exists()

            if modality in ["video", "multimodal"]:
                ok = ok and (
                    conv / "features" / "video" / "speaker_0.facenet.parquet"
                ).exists()
                ok = ok and (
                    conv / "features" / "video" / "speaker_1.facenet.parquet"
                ).exists()

            if ok:
                keep.append(i)

        self.labels = self.labels.iloc[keep].reset_index(drop=True)

        if len(self.labels) == 0:
            self.input_dim = 0
            self.audio_dim = None
            self.video_dim = None
            return

        sample = self.labels.iloc[0]
        conv = self.data_root / sample["conversation_id"]

        self.audio_dim = None
        self.video_dim = None

        if modality in ["audio", "multimodal"]:
            _, emb = _load_audio(
                str(conv / "features" / "audio" / "whisper_small.parquet")
            )
            self.audio_dim = emb.shape[1]

        if modality in ["video", "multimodal"]:
            _, emb = _load_video(
                str(conv / "features" / "video" / "speaker_0.facenet.parquet")
            )
            self.video_dim = emb.shape[1]

        self.input_dim = (
            self.audio_dim
            if modality == "audio"
            else self.video_dim * 2
            if modality == "video"
            else self.audio_dim + self.video_dim * 2
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        conv = self.data_root / row["conversation_id"]
        start_t, end_t = float(row["start_sec"]), float(row["end_sec"])

        feats = []

        if self.modality in ["audio", "multimodal"]:
            ts, emb = _load_audio(
                str(conv / "features" / "audio" / "whisper_small.parquet")
            )
            feats.append(_pool(ts, emb, start_t, end_t, self.audio_dim))

        if self.modality in ["video", "multimodal"]:
            for spk in ["speaker_0", "speaker_1"]:
                ts, emb = _load_video(
                    str(conv / "features" / "video" / f"{spk}.facenet.parquet")
                )
                feats.append(_pool(ts, emb, start_t, end_t, self.video_dim))

        x = np.nan_to_num(np.concatenate(feats).astype(np.float32))
        y = np.float32(row["rapport_score"])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def collate(batch):
    return (
        torch.stack([b["x"] for b in batch]),
        torch.stack([b["y"] for b in batch]),
        torch.stack([b["index"] for b in batch]),
    )


class RapportDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.test_dataset = None

    def setup(self, stage=None):
        labels = pd.read_csv(self.config["data"]["labels_file"])
        kwargs = dict(
            data_root=self.config["data"]["data_root"],
            modality=self.config["data"]["modality"],
        )

        train_df = labels[labels["split"] == "train"]
        val_df = labels[labels["split"] == "val"]
        test_df = (
            labels[labels["split"] == "test"]
            if "test" in set(labels["split"].astype(str))
            else labels.iloc[0:0]
        )

        self.train_dataset = RapportDataset(train_df, **kwargs)
        self.val_dataset = RapportDataset(val_df, **kwargs)
        self.test_dataset = RapportDataset(test_df, **kwargs) if len(test_df) > 0 else None
        self.input_dim = self.train_dataset.input_dim

        print(f"Modality:      {self.config['data']['modality']}")
        print(
            f"Train windows: {len(self.train_dataset)} "
            f"({self.train_dataset.labels['conversation_id'].nunique()} conversations)"
        )
        print(
            f"Val windows:   {len(self.val_dataset)} "
            f"({self.val_dataset.labels['conversation_id'].nunique()} conversations)"
        )

        test_n = len(self.test_dataset) if self.test_dataset is not None else 0
        test_c = (
            self.test_dataset.labels["conversation_id"].nunique()
            if self.test_dataset is not None
            else 0
        )

        print(f"Test windows:  {test_n} ({test_c} conversations)")
        print(f"Input dim:     {self.input_dim}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=collate,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=collate,
            pin_memory=False,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test labels are not available in this release.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=collate,
            pin_memory=False,
        )