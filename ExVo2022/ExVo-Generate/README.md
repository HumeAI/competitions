# ExVo Generate
For this track participants should develop a generative model to generate 1 to 10 classes of the emotionally expressive vocal bursts.

Also submission for this track should include the calculation of the Fréchet Inception Distance (FID) between generated and original files. 

This repository contains:
- Starter code for loading the ExVo data into the baseline model (found in `loaders`)
- A script to compute FID using a pretrained (on Hume-VB) classifier model (inspired by https://github.com/mseitzer/pytorch-fid)

## Setup

In a new Python virtual environment
```
pip install --upgrade pip
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu11pip0 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data Loader Usage

Please see the [NFB](https://github.com/nfb-onf/sound-of-laughter) repository for the model applied. To train using the ExVo dataset, the main training loop `loaders/train.py` can be modified as a starting point for your own generative training.

The csv file for training is provided with the data package, `data_info.csv`.

```
python loaders/train.py --data_path <wav_path> --csv_path <path_to_csv> --emotion <emotion_name or All> --country <country_name or All>

# For example, to train on everything
python loaders/train.py --data_path <wav_path> --csv_path <path_to_csv> --emotion All --country All

# To only train on Amusement samples from the United States
python loaders/train.py --data_path <wav_path> --csv_path <path_to_csv> --emotion Amusement --country United States
```

## Calculating FID between original and generated samples
Included in `fid/` are precomputed gaussian statistics for the validation sets of each emotion (as well as one for the whole dataset). The `_wav.pkl` and  `_webm.pkl` are provided. 

Additionally, we provide 2 helper scripts. `save_statistics.py` can be used to save the activation statistics (i.e. the gaussian statistics) of the trained model run on a folder of `.wav` samples:
```
# The below saves the activation statistics of the .wav samples in <wav_path> in fid/<name>.pkl
python save_statistics.py --samples_1 <wav_path> --name <name>
```

To compute FID between 2 sets of samples, use the `fid.py` script:
```
# Between 2 folders of .wav samples
python fid.py --samples_1 <path_to_folder_1> --samples_2 <path_to_folder_2>

# Between a folder of .wav samples and precomputed statistics in <name.pkl
python fid.py --samples_1 <path_to_folder_1> --samples_2 <name>.pkl

# For example, the below computes the fid of the generated samples and the precomputed Amusement validation activations:
python fid.py --samples_1 ./wav_samples/amusement --samples_2 Amusement.pkl
```

[update 10 May 2022]: We have updated the repository to include the model and precomputed statistics based on the recently distributed `webm` data. For reproducability, we also include the script to calculate the gaussian statistics `save_statistics.py`. 





