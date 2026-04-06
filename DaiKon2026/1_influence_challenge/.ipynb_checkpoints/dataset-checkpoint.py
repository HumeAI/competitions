from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


EMOTIONS = [
    "anger",
    "anxiety",
    "uncertainty",
    "confusion",
    "doubt",
    "boredom",
    "surprise",
    "curiosity",
    "joy",
    "amusement",
]


@lru_cache(maxsize=32)
def _load_audio(path: str):
    df = pq.read_table(path).to_pandas()
    if len(df) == 0:
        return None, None

    timestamps = df["timestamp"].to_numpy(dtype=np.float32)
    embeddings = np.stack(df["embedding"].values).astype(np.float32)
    return timestamps, embeddings


@lru_cache(maxsize=64)
def _load_video(path: str):
    df = pq.read_table(path).to_pandas()
    if len(df) == 0:
        return None, None

    timestamps = df["timestamp"].to_numpy(dtype=np.float32)
    embeddings = np.stack(df["embedding"].values).astype(np.float32)
    return timestamps, embeddings


def _pool(ts, emb, start_t, end_t, fallback_dim=None):
    if ts is None or emb is None:
        return np.zeros((fallback_dim,), dtype=np.float32)

    idx = np.where((ts >= start_t) & (ts <= end_t))[0]
    if len(idx) == 0:
        center_t = 0.5 * (start_t + end_t)
        idx = np.array([int(np.argmin(np.abs(ts - center_t)))])

    pooled = emb[idx].mean(axis=0).astype(np.float32)
    return np.nan_to_num(pooled)


class InfluenceDataset(Dataset):
    def __init__(self, labels_df, data_root, modality="audio"):
        self.labels = labels_df.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.modality = modality

        keep = []
        original_len = len(self.labels)

        for i, row in self.labels.iterrows():
            conv = self.data_root / row["conversation_id"]
            ok = True

            if modality in ["audio", "multimodal"]:
                ok = ok and (conv / "features" / "audio" / "whisper_small.parquet").exists()

            if modality in ["video", "multimodal"]:
                ok = ok and (conv / "features" / "video" / "speaker_0.facenet.parquet").exists()
                ok = ok and (conv / "features" / "video" / "speaker_1.facenet.parquet").exists()

            if ok:
                keep.append(i)

        self.labels = self.labels.iloc[keep].reset_index(drop=True)

        print(
            f"[InfluenceDataset] modality={modality} | "
            f"input_rows={original_len} | "
            f"kept_rows={len(self.labels)} | "
            f"dropped_rows={original_len - len(self.labels)}",
            flush=True,
        )

        self.audio_dim = None
        self.video_dim = None
        self.input_dim = 0

        if len(self.labels) == 0:
            return

        sample = self.labels.iloc[0]
        conv = self.data_root / sample["conversation_id"]

        if modality in ["audio", "multimodal"]:
            _, emb = _load_audio(str(conv / "features" / "audio" / "whisper_small.parquet"))
            self.audio_dim = emb.shape[1]

        if modality in ["video", "multimodal"]:
            _, emb = _load_video(str(conv / "features" / "video" / "speaker_0.facenet.parquet"))
            self.video_dim = emb.shape[1]

        if modality == "audio":
            self.input_dim = self.audio_dim
        elif modality == "video":
            self.input_dim = self.video_dim * 2
        else:
            self.input_dim = self.audio_dim + self.video_dim * 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        conv = self.data_root / row["conversation_id"]
        speaker_id = row["speaker_id"]
        partner_id = "speaker_1" if speaker_id == "speaker_0" else "speaker_0"

        start_t = float(row["start_sec"])
        end_t = float(row["end_sec"])

        feats = []

        if self.modality in ["audio", "multimodal"]:
            ts, emb = _load_audio(str(conv / "features" / "audio" / "whisper_small.parquet"))
            feats.append(_pool(ts, emb, start_t, end_t, self.audio_dim))

        if self.modality in ["video", "multimodal"]:
            ts, emb = _load_video(str(conv / "features" / "video" / f"{speaker_id}.facenet.parquet"))
            feats.append(_pool(ts, emb, start_t, end_t, self.video_dim))

            ts, emb = _load_video(str(conv / "features" / "video" / f"{partner_id}.facenet.parquet"))
            feats.append(_pool(ts, emb, start_t, end_t, self.video_dim))

        x = np.nan_to_num(np.concatenate(feats).astype(np.float32))
        y = np.nan_to_num(row[EMOTIONS].to_numpy(dtype=np.float32))

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def collate(batch):
    return (
        torch.stack([b["x"] for b in batch]),
        torch.stack([b["y"] for b in batch]),
        torch.stack([b["index"] for b in batch]),
    )


class InfluenceDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        labels = pd.read_csv(self.config["data"]["labels_file"])

        kwargs = {
            "data_root": self.config["data"]["data_root"],
            "modality": self.config["data"]["modality"],
        }

        self.train_dataset = InfluenceDataset(
            labels[labels["split"] == "train"],
            **kwargs,
        )
        self.val_dataset = InfluenceDataset(
            labels[labels["split"] == "val"],
            **kwargs,
        )

        has_test = "test" in set(labels["split"].astype(str))
        test_df = labels[labels["split"] == "test"] if has_test else labels.iloc[0:0]

        self.test_dataset = (
            InfluenceDataset(test_df, **kwargs)
            if len(test_df) > 0
            else None
        )

        if len(self.train_dataset) == 0:
            raise ValueError("Train dataset is empty after filtering for available features.")
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty after filtering for available features.")
        if self.test_dataset is not None and len(self.test_dataset) == 0:
            raise ValueError("Test dataset is empty after filtering for available features.")

        self.input_dim = self.train_dataset.input_dim

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