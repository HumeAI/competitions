# ACII DaiKon Challenge Baselines

This bundle includes simple baselines for the three ACII DaiKon workshop challenges:

1. **Influence** — predict 10 continuous emotion intensities for each labeled segment.
2. **Turn-Taking** — predict whether the next speaker changes and when the next speech onset occurs.
3. **Rapport** — predict a continuous rapport score for each labeled conversation window.

## Environment

Create and activate the conda environment:

```bash
conda create -n daikon-challenges python=3.11 -y
conda activate daikon-challenges
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

To update it later:

```bash
conda env update -f environment.yml --prune
conda activate daikon-challenges
```

## Expected public dataset layout

```text
data/
├── conv_000001/
│   ├── features/
│   │   ├── audio/
│   │   │   └── whisper_small.parquet
│   │   └── video/
│   │       ├── speaker_0.facenet.parquet
│   │       └── speaker_1.facenet.parquet
├── labels_rapport/
│   ├── daikon_rapport_train.csv
│   └── daikon_rapport_val.csv
├── labels_turn_taking/
│   ├── daikon_turn_taking_train.csv
│   └── daikon_turn_taking_val.csv
└── labels_influence/
    ├── daikon_influence_train.csv
    └── daikon_influence_val.csv
```

## Challenge 1: Influence

Goal: predict 10 continuous emotion intensities for each labeled segment.

Train:
```bash
cd daikon_influence_baseline
python prepare_public_labels.py
python -u train.py --config config.yaml --seed 42 --no-wandb
```

Evaluate:
```bash
python evaluate.py --run-dir /path/to/run_dir --split test
```

Modes:
- `audio`
- `video`
- `multimodal`

Metrics:
- mean CCC
- mean Pearson
- per-emotion CCC / Pearson

## Challenge 2: Turn-Taking

Goal: predict:
1. whether the next speaker changes
2. the time until the next speech onset

Train:
```bash
cd daikon_turn_taking_baseline
python -u train.py --config config.yaml --seed 42 --no-wandb
```

Evaluate:
```bash
python evaluate.py --run-dir /path/to/run_dir --split test
```

Modes:
- `audio`
- `video`
- `multimodal`

Metrics:
- Macro-F1
- Accuracy
- MAE

## Challenge 3: Rapport

Goal: predict a continuous rapport score for each labeled conversation window.

Train:
```bash
cd daikon_rapport_baseline
python -u train.py --config config.yaml --seed 42 --no-wandb
```

Evaluate:
```bash
python evaluate.py --run-dir /path/to/run_dir --split test
```

Modes:
- `audio`
- `video`
- `multimodal`

Metrics:
- CCC
- Pearson
- MAE

## Notes

- Audio baselines use pooled Whisper Small embeddings.
- Video baselines use pooled FaceNet embeddings for both speakers.
- Multimodal baselines concatenate pooled audio and pooled video features.
