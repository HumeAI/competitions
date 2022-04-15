# ExVo FewShot

For this sub-challenge you are tasked to perform personalized recognition of emotional vocalizations using few-shot learning. The baseline code trains a model without leveraging the speaker's id information but during evaluation the speaker's id are considered. The code is provided in this repo for participants to reproduce the baseline.

## Installation

The required packages to run this code can be installed by

```
pip install -r requirements.txt
```

## Training

Before starting training the required csv files need to be created. This can be performed with the following `create_splits.py` scirpt.

```
python create_scripts.py --data_file_path=/path/to/data_info.csv \
                         --save_path=/path/to/save/csv/files
```

After the csv files have been created the training of the model can start. This is performed by the `perform_training.py` script.

```
python perform_training.py --csv_paths=/path/to/csv/files \
                           --data_path=/path/to/wav/folder \
                           --base_dir=/path/to/save/models \
                           --learning_rate=0.001 \
                           --batch_size=8 \
                           --number_of_epochs=60
```

## Few-Shot Evaluation

When the speakers id for the test set becomes available you need to uncommnent the last lines in the `create_splits.py` script so that the json files required for evaluation are created. Then you can start the evaluation.

```
python perform_evaluation.py --json_path=/path/to/json/file \
                             --data_path=/path/to/wav/folder 
```

You can also try to use the validation as a test set to evaluate your model. In more detail, you can split the training set to new train/valid splits, train a model on these splits, and then use the validation for test purposes.

## Submitting predictions 

Please submit your results in the following format.

`ExVo-Few_<team_name>_<submission_no>.csv` 

|File_ID|Awe   |Excitement|Amusement|Awkwardness|Fear   |Horror|Distress|Triumph|Sadness|Surprise|
|-------|------|----------|---------|-----------|-------|------|--------|-------|-------|--------|
|[58712]|0.109 |0.258     | 0.159   |0.181      |0.605  |0.591 |0.462   |0.071  |0.160  |0.520   |

Each team will have 5 submission oppurtunties to submit their predictions to `competitions@hume.ai`
