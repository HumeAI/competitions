# ExVo MultiTask

For this track participants should develop a multitask model to jointly learn from 10-classes of emotional expression, as well as age, and native-country. In this repo the baseline multitask model is provided for participants to reproduce.

## Installation

First make sure you download the data and features, placing them within your working directory. For more info and instructions on how to access the competition data visit [competitions.hume.ai](http://www.competitions.hume.ai). 

We suggest creating a virtual enviroment, and installing the `requirements.txt`

```
pip install -r requirements.txt
```

## Example

After the data, labels, and features are downloaded to your working directory, when running `main.py` set to `-d ../` i.e., where `feats/` and `data_info.csv` is.

_Baseline_

```
  python3 main.py -d ../ -l data_info.csv -f eGeMAPS -tn Baseline --store_pred --save_csv --n_seeds 5 --pltloss
```

| Option         | Description                                  |
| -------------- | -------------------------------------------- |
| `-d`           | Set the path to working dir                  |
| `-l`           | Label file e.g., `data_info.csv`             |
| `-f`           | Feature set e.g., `eGeMAPS`                  |
| `-bs`          | Batch Size  (default: `8`)                   |
| `-lr`          | Learning Rate  (default: `0.001`)            |
| `-e`           | Number of Epochs (default: `20`)             |
| `-p`           | Early Stopping Patience (default: `5`)       |
| `-tn`          | Team Name (default: `Baseline`)              |
| `--store_pred` | Store Predictions (Action: Store `True`)     |
| `--save_csv`   | Save Results as `csv` (Action: Store `True`) |
| `--pltloss`    | Save Loss plot                               |
| `--n_seeds`    | Number of Seeds to run for                   |



## Submitting predictions 

In `main.py`, `store_predictions()` can be used to create the needed predicitions file for this task. You should submit a comma seperated csv file: 

`ExVo-Multi_<team_name>_<submission_no>.csv` 

|File_ID|Country|Age  |Awe   |Excitement|Amusement|Awkwardness|Fear   |Horror|Distress|Triumph|Sadness|Surprise|
|-------|-------|-----|------|----------|---------|-----------|-------|------|--------|-------|-------|--------|
|[58712]| 1     |32.2 |0.109 |0.258     | 0.159   |0.181      |0.605  |0.591 |0.462   |0.071  |0.160  |0.520   |

Each team will have 5 submission oppurtunties to submit their predictions to `competitions@hume.ai`
