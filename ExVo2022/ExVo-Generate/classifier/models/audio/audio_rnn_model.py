import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .audio_model import AudioModel
from ..rnn import RNN


class AudioRNNModel(nn.Module):
    
    def __init__(self,
                 input_size:int,
                 num_outs:int):
        """ Convolutional recurrent neural network model.
        
        Args:
            input_size (int): Input size to the model. 
            num_outs (int): Number of output values of the model.
        """
        
        super(AudioRNNModel, self).__init__()
        audio_network = AudioModel(input_size=input_size)
        self.audio_model = audio_network.model
        num_out_features = audio_network.num_features
        self.rnn, r1_num_out_features = self._get_rnn_model(num_out_features)
        self.linear = nn.Linear(r1_num_out_features, num_outs)
        
        self.num_outs = num_outs
    
    def _get_rnn_model(self, input_size:int):
        """ Builder method to get RNN instace."""
        
        rnn_args = {
            'input_size': input_size,
            'hidden_size': 256,
            'num_layers': 2,
            'batch_first': True
        }
        
        return RNN(rnn_args, 'lstm'), rnn_args['hidden_size']
    
    def forward(self, x:torch.Tensor):
        """
        Args:
            x ((torch.Tensor) - BS x S x x T)
        """
        
        batch_size, seq_length, t = x.shape
        x = x.view(batch_size*seq_length, 1, t)
        
        audio_out = self.audio_model(x)
        audio_out = audio_out.view(batch_size, seq_length, -1)
        
        rnn_out, _ = self.rnn(audio_out)

        return rnn_out
        # reshaped_rnn = rnn_out.reshape(batch_size*seq_length, -1)
        # output = self.linear(reshaped_rnn)
        #
        # return output.view(batch_size, seq_length, -1)
