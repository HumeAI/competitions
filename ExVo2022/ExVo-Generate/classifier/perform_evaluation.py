import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
import json

from evaluation import Evaluation 
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument('--files_path', type=str, default='./',
                    help="Path to csv files.")
parser.add_argument('--ext', type=str, default='mp3',
                    help="Extention of the files to test model (default wav).")
parser.add_argument('--save_preds_path', type=str, default='./',
                    help="Directory to save model/logs (default ./).")
parser.add_argument('--batch_size', type=int, default=1,
                    help="Batch size to use for the experiment (default 1).") 
parser.add_argument('--use_gpu', type=int, default=0,
                    help="Whether to use GPU (default True).") 
parser.add_argument('--model_path', type=str, help="Path to model.")


def main(dataset_params, eval_params):
    evaluation = Evaluation(dataset_params, eval_params)
    evaluation.start()

if __name__ == '__main__':
    args = parser.parse_args()
    
    dataset_params = {
        'batch_size': 32,
        'test':{'files_path': "/network/datasets/restricted/icmlexvo2022_users/icmlexvo2022.var/icmlexvo2022_extract/wav", 'ext': "wav"}
    }
    
    use_gpu = False if args.use_gpu == 0 else True
    
    eval_params = {
        'use_gpu': True,
        'save_preds_path': "",
        'model_path': "/home/mila/m/marco.jiralerspong/projects/sound-of-laughter/classifier/pretrained_model/predictions_model.pt",
    }
    
    main(dataset_params, eval_params)
