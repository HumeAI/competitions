import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import json

from argparse import Namespace
from pathlib import Path
from torch.utils.data import DataLoader

from classifier.data_provider import get_dataloader
from classifier.models import AudioRNNModel


class Evaluation:
    
    def __init__(self,
                 dataset_params,
                 eval_params):
        """ Performs the evaluation of the model.
        
        Args:
          dataset_params  (dict): the parameters to load the daset.
              'test'      (dict): Arguments to the Dataset class.
              'batch_size' (int): Batch size to use during training.
                    
          eval_params (dict): Training parameters.
                'model_path'       (str)  : The directory to save best model/logs.
                'use_gpu'          (bool) : Whether to use GPU.
        """
        
        self.dataset = get_dataloader(
            dataset_params['batch_size'], shuffle=False, **dataset_params['test'])

        num_input_features = self.dataset.dataset.num_input_features
        num_input_features = 1600
        self.model = AudioRNNModel(num_input_features, 10)

        
        self.eval_params = Namespace(**eval_params)
        self.device = torch.device('cuda:0') if self.eval_params.use_gpu else torch.device('cpu')

        if self.eval_params.use_gpu:
            self.model.cuda()

        self.load_model()
        
    def load_model(self):
        """Loads model parameters (state_dict) from file_path."""
        
        print(f'Loading model parameter from {str(self.eval_params.model_path)}')
        
        if not Path(self.eval_params.model_path).exists():
            raise Exception(f'No model exists in path : {str(self.eval_params.model_path)}')
        checkpoint = torch.load(str(self.eval_params.model_path), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint
    
    def start(self):
        self.perform_eval_epoch(self.dataset)
        
    def save_predictions(self, predictions):
        
        print('Saving Predictions...')
        
        save_preds_path = Path(self.eval_params.save_preds_path) / 'predictions.json'
        
        with open(str(save_preds_path), 'w') as f:
            json.dump(predictions, f, indent=2)

    def perform_eval_epoch(self, dataset:DataLoader):
        
        self.model.eval()
        
        preds = {}
        
        for n_iter, (batch_data, mask, filenames) in enumerate(dataset):
            print(f'Iter:{n_iter+1}/{len(self.dataset)}')
            
            if self.eval_params.use_gpu:
                batch_data = batch_data.cuda()
            
            predictions = self.model(batch_data)
            
            # Get model predictions
            for i, (f, m) in enumerate(zip(*[filenames, mask])):
                pred_f = predictions[i,m-1].data.cpu().numpy()
                preds[f] = pred_f.astype(np.float32).tolist()
                
        self.save_predictions(preds)
