import torch
import torch.nn as nn

from .emo18 import Emo18


class AudioModel(nn.Module):
    
    def __init__(self, 
                 *args, **kwargs):
        """ Audio network model.
        
        Args:
            model_name (str): Name of audio model to use.
            pretrain (bool): Whether to use pretrain model (default `False`).
        """
        
        super(AudioModel, self).__init__()
        
        self.model = Emo18(*args, **kwargs)
        self.num_features = self.model.num_features
            
    def forward(self, x):
        """ Forward pass
        
        Args:
            x (BS x S x 1 x T)
        """
        return self.model(x)
