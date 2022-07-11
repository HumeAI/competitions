
# A-VB Feature-based

We provide a feature-based approach for all four tasks for participants to reproduce the official competition baselines. 

## Installation

First make sure you download the data and features, placing them within your working directory. For more info and instructions on how to access the competition data visit [competitions.hume.ai](http://www.competitions.hume.ai). 

We suggest creating a virtual environment, and installing the `requirements.txt`

```
pip install -r requirements.txt
```

## Example

After the data, labels, and features are downloaded to your working directory, when running `main.py` set to `-d ./` i.e., where `features/` and `labels/` are, and run: 

_A-VB High Baseline_

```
   python main.py -d ./ -f eGeMAPS -t high -e 25 -lr 0.001 -bs 8 -p 5 --n_seeds 5
```

Follow the same procedure for each task altering `-t`. 


| Option         | Description                                  |
| -------------- | -------------------------------------------- |
| `-d`           | Set the path to working dir                  |
| `-f`           | Feature set e.g., `eGeMAPS`                  |
| `-t`           | Task ['high','two','culture','type']         |
| `-e`           | Number of Epochs (default: `20`)             |
| `-lr`          | Learning Rate  (default: `0.001`)            |
| `-bs`          | Batch Size  (default: `8`)                   |
| `-p`           | Early Stopping Patience (default: `5`)       |
| `--n_seeds`    | Number of Seeds to run for                   |
| `--verbose`    | Maximum verbosity, (default: quiet)          |
