# Rapport Trajectory Sub-Challenge Baseline

Baseline for the ACII DaiKon Rapport sub-challenge: predicting how rapport evolves over time in dyadic conversation.

## Task

For each labeled conversation window, predict a **continuous rapport score**.

## Data

- **Input**: Released feature files from the public dataset
  - Audio: Whisper Small embeddings from `features/audio/whisper_small.parquet`
  - Video: FaceNet embeddings from `features/video/speaker_0.facenet.parquet` and `speaker_1.facenet.parquet`
- **Labels**: Window-level rapport trajectory labels from `daikon_rapport_labels_all.csv`
- **Splits**: Read from the shared `splits.json` / label split column

For each labeled window `[start_sec, end_sec)`, the baseline mean-pools features over the window.

## Modes

The same codebase supports three settings via `config.yaml`:

- `modality: audio`
- `modality: video`
- `modality: multimodal`

## Architecture

```
Window-level pooled feature vector
    -> Shared MLP encoder:
         Linear(input_dim, 256) -> ReLU -> Dropout
         Linear(256, 256) -> ReLU -> Dropout
         Linear(256, 1)
    -> rapport score
```

## Evaluation

- **Rapport**: CCC, Pearson correlation, and MAE

## Usage

### Label Preparation (organizers/internal, optional)

```bash
python prepare_labels.py \
  --raw-labels /mnt/weka/panos/projects/acii_daoikon_workshop/rapport_out/all_pseudolabels.csv \
  --mapping-csv /mnt/weka/acii_daikon_workshop/data/_PRIVATE_mapping.csv \
  --splits-csv /mnt/weka/acii_daikon_workshop/data/splits/splits.csv \
  --output-dir /mnt/weka/acii_daikon_workshop/data/labels
```

### Training

```bash
export WANDB_API_KEY=<your-key>  # optional
python train.py --config config.yaml --seed 42 [--no-wandb]
```

### Evaluation

```bash
python evaluate.py --run-dir /path/to/run_dir --split test
```

## File Structure

```
rapport_baseline/
├── config.yaml
├── prepare_labels.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── utils.py
└── README.md
```

## Notes

- This is an intentionally simple baseline.
- Each sample is one labeled rapport window.
- Audio/video features are pooled independently over the window.
- Participants can replace the mean pooling with temporal models, attention, or sequence encoders.
