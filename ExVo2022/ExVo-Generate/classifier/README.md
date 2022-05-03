Emotion predictions of vocal bursts.

### Predicting on Audio Data
The arguments of the script are the following:

| Flag | Description | Type | Default |
| :---: | :---: | :---: | :---: |
| --files_path | Path to the directory of audio files. | str | - |
| --ext | Extension of the files to predict on. | str | wav |
| --batch_size | The batch size to use. | int | 1 |
| --use_gpu | Whether to use GPU. Use 1 or 0. | int | 1 |
| --model_path | Path to model weights. | str | - |
| --save_preds_path | Path to save the predictions (`predictions.json`) of the model to. | str | ./ |

An example on how to run this script shown below.

> Start evaluation of the model
```bash
python perform_evaluation.py --files_path=/path/to/folder \
                             --model_path=./pretrained_model/predictions_model.pt \
                             --ext=wav \ 
                             --batch_size=3
```

The output is a `json` file with the `key` to be the name of the files evaluated, and `values` the corresponding predictions of the model.
