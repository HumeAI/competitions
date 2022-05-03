import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

from .audio import AudioRNNModel
from torchvision import transforms


def get_model(network_name, *args, **kwargs):
    return {
        'melspec': MelSpecNet,
        'e2e': AudioRNNModel
    }[network_name](*args, **kwargs)


class MelSpecNet(nn.Module):
    
    def __init__(self,
                 input_size:int = None,
                 video_net:str = 'resnet18',
                 num_classes:int = 6):
        """ Creates the network architecture.
        
        Args:
          video_net : The name of the network to use.
        """
        
        super(MelSpecNet, self).__init__()
        
        network, fc_inp_size = self._get_model_and_out_size(video_net)
        network = list(network.children())[:-1]
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.network = nn.Sequential(*network)
        
        self.preds = nn.Linear(fc_inp_size, num_classes)
        self.num_classes = num_classes
    
    def _get_model_and_out_size(self, video_net:str):
        (network, num_out_features) = {
            'resnet18':(models.resnet18, 512),
            'resnet34':(models.resnet34, 512),
            'resnet50':(models.resnet50, 2048),
            'vgg11':(models.vgg11, 25088),
            'vgg13':(models.vgg13, 25088),
            'vgg16':(models.vgg16, 25088),
        }[video_net]
        
        return network(pretrained=True), num_out_features
    
    def forward(self, x:torch.tensor):
        bs, sl, c = x.shape
        
        # bs, c, sl
        x = x.permute(0,2,1)
        x = x.view(bs, 3, 128, sl)
        
        x = self.normalize(x)
        
        cnn_out = self.network(x)
        cnn_out = cnn_out.view(bs, -1)
        
        output = self.preds(cnn_out)
        
        return output


class CNN(nn.Module):
    
    def __init__(self, 
                 input_size:int,
                 op_params:dict,
                 mp_params:dict,
                 rnn_params:dict,
                 num_classes:int = 6,
                 act_fn_name:str = 'leaky_relu'):
        """ Creates the network architecture.
        
        Args:
          input_size (int): The input dimensionality of the data.
          hidden_unit_layers (list): List with the number of neurons at each layer.
          num_classes (int): Number of output classes to predict.
          act_fn (str): Activation function to use in the network.
        """
        
        super(CNN, self).__init__()
        act_fn = nn.LeakyReLU()
        num_layers = len(op_params.keys())
        
        layers = nn.ModuleList([])
        channel_size = 1
        layer_input_size = input_size
        for op_par, mp_par in zip(*[op_params.values(), mp_params.values()]):
            layers.extend(
                self.cnn_block(act_fn, op_par, mp_par)
            )
            # Compute convolution output
            num_out_feats = np.floor(
                (layer_input_size - op_par['kernel_size'] + 2*op_par['padding']) + 1)
            
            # Compute max pooling output
            num_out_feats = np.floor((num_out_feats - mp_par['kernel_size'])/mp_par['stride']) + 1
            layer_input_size = num_out_feats
        
        self.cnn_net = nn.Sequential(*layers)
        
        rnn_params['input_size'] = int(layer_input_size*op_par['out_channels'])
        self.rnn_net = nn.GRU(**rnn_params)
        
        self.num_outs = num_outs
        fc_inp_size = rnn_params['hidden_size']
        self.preds = nn.Linear(fc_inp_size, num_outs)
        
        self.reset_parameters()
    
    def cnn_block(self, act_fn, op_params, mp_params):
        return [nn.Conv1d(**op_params),
                nn.BatchNorm1d(op_params['out_channels']),
#                 nn.Dropout(p=0.2),
                act_fn,
                nn.MaxPool1d(**mp_params)]
    
    def reset_parameters(self):
        """ Initialize parameters of the model."""
        for m in list(self.modules()):
            self._init_weights(m)
    
    def _init_weights(self, m):
        """ Helper method to initialize the parameters of the model 
            with Kaiming uniform initialization.
        """
        
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        if type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(param)
    
    def forward(self, x:torch.tensor):
        '''
            x (batch_size, seq_len, nchannels, nfeatures)
        '''
        bs, sl, c, t = x.shape
        
        x = x.view(-1, c, t)
        output = self.cnn_net(x)
        
        cnn_out = output.view(bs, sl, -1)
        
        rnn_out, _ = self.rnn_net(cnn_out)
        
        output = self.preds(rnn_out.reshape(bs*sl, -1))
        
        return output.view(bs, sl, -1)
