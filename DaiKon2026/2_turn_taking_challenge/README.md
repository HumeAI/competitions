# Turn-Taking Sub-Challenge Baseline

Baseline for the ACII DaiKon Turn-Taking sub-challenge: predicting who speaks next and when in dyadic conversation.

## Task

At each prediction point in a dyadic conversation, predict:

1. **Next-speaker change** (binary classification): does the same speaker continue (`0`) or does the other speaker take the turn (`1`)?
2. **Time-to-next-speech** (regression): how many seconds until the next speech onset? Can be negative for overlaps.

## Data

- **Input**: Released feature files from the public dataset
  - Audio: `features/audio/whisper_small.parquet`
  - Video: `features/video/speaker_0.facenet.parquet` and `features/video/speaker_1.facenet.parquet`
- **Labels**: Public turn-taking labels from:
  - `/mnt/weka/acii_daikon_workshop/data/labels_turn_taking/daikon_turn_taking_labels_all.csv`
- **Splits**: Read from the `split` column in the label CSV

For each labeled prediction point, the baseline uses a fixed context window before `prediction_time`.

## Modes

Set `data.modality` in `config.yaml` to one of:

- `audio`
- `video`
- `multimodal`

## Architecture

```text
Context-window pooled feature vector
    -> Shared MLP encoder:
         Linear(input_dim, 256) -> ReLU -> Dropout
         Linear(256, 256) -> ReLU -> Dropout
    -> Speaker head: Linear(256, 1)
    -> Time head: Linear(256, 1)
```

## Evaluation

- **Next-speaker**: Macro-F1 and accuracy
- **Time-to-next-speech**: MAE

## Usage

### Training

```bash
python -u train.py --config config.yaml --seed 42 --no-wandb
```

### Evaluation

```bash
python evaluate.py --run-dir /path/to/run_dir --split test
```

## Notes

- This is an intentionally simple baseline.
- Audio uses mean-pooled Whisper Small embeddings over the context window.
- Video uses mean-pooled FaceNet embeddings for both speakers over the context window.
- Multimodal concatenates pooled audio + pooled video features.
