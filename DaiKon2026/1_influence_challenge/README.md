# Influence Sub-Challenge Baseline

Baseline for the ACII DaiKon Influence sub-challenge: predicting per-segment emotion intensities for a target speaker.

## Task

For each labeled speech segment, predict 10 continuous emotion intensity scores:

- anger
- anxiety
- uncertainty
- confusion
- doubt
- boredom
- surprise
- curiosity
- joy
- amusement

## Data

- **Input**: Released participant-facing features
  - Audio: `features/audio/whisper_small.parquet`
  - Video: `features/video/speaker_0.facenet.parquet` and `features/video/speaker_1.facenet.parquet`
- **Labels**: Public influence labels produced by `prepare_public_labels.py`
- **Splits**: Read from the `split` column in the label CSV

## Modes

Set `data.modality` in `config.yaml` to one of:

- `audio`
- `video`
- `multimodal`

## Architecture

A simple mean-pooled baseline:

```text
Segment pooled feature vector
    -> Shared MLP encoder:
         Linear(input_dim, 256) -> ReLU -> Dropout
         Linear(256, 256) -> ReLU -> Dropout
    -> Linear(256, 10) -> Sigmoid
```

## Evaluation

- Mean CCC across the 10 emotions
- Mean Pearson across the 10 emotions
- Per-emotion CCC and Pearson

## Usage

### Prepare participant-facing labels

```bash
python prepare_public_labels.py
```

### Training

```bash
python -u train.py --config config.yaml --seed 42 --no-wandb
```

### Evaluation

```bash
python evaluate.py --run-dir /path/to/run_dir --split test
```

## Notes

- Audio uses mean-pooled Whisper Small embeddings over the target segment.
- Video uses mean-pooled FaceNet embeddings for the target speaker and partner speaker over the target segment.
- Multimodal concatenates pooled audio + pooled target-speaker video + pooled partner video.
