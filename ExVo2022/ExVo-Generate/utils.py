import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import os

from frechet_distance import calculate_frechet_distance
from classifier.data_provider import get_dataloader
from classifier.models import AudioRNNModel

MODEL_PATH = "./classifier/pretrained_model/predictions_model_webm.pt"

def load_model(device):
    """Loads model parameters (state_dict) from file_path."""

    print(f'Loading model parameter from {MODEL_PATH}')

    model = AudioRNNModel(1600, 10)

    if not Path(MODEL_PATH).exists():
        raise Exception(f'No model exists in path : {MODEL_PATH}')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_activation_statistics(name):
    base_path = "./fid/"
    path = os.path.join(base_path, name)
    with open(path, "rb") as f:
        dict = pickle.load(f)
        mu = dict["mu"]
        sigma = dict["sigma"]
    return mu, sigma

def compute_activation_statistics(preds):
    mu = np.mean(preds, axis=0)
    sigma = np.cov(preds, rowvar=False)
    return mu, sigma

def get_activation_statistics(path, batch_size):
    dataset_params = {
        "files_path": path,
        "ext": "wav",
    }

    dataset = get_dataloader(batch_size, shuffle=False, **dataset_params)
    num_samples = len(dataset.dataset.files_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device)
    model = model.to(device)
    model.eval()

    preds = np.zeros((num_samples, 256))

    for n_iter, (batch_data, mask, filenames) in tqdm(enumerate(dataset)):
        batch_data = batch_data.cuda()
        predictions = model(batch_data)
        for sample in range(batch_size):
            curr = n_iter * batch_size + sample
            if curr >= num_samples:
                break

            preds[curr] = predictions[sample, mask[sample]-1].detach().cpu().numpy()

    mu, sigma = compute_activation_statistics(preds)
    return mu, sigma

def get_fid(name1, name2):
    mu1, sigma1 = load_activation_statistics(name1)
    mu2, sigma2 = load_activation_statistics(name2)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid