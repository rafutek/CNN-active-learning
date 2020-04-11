# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn


class CNNBaseModel(nn.Module):
    """
    Base class for all CNN models
    """

    def __init__(self, num_classes=10, init_weights=True):
        """
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(CNNBaseModel, self).__init__()
        self.num_classes = num_classes
        if init_weights:
            self._initialize_weights()

    def forward_layer(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        raise NotImplementedError

    def _initialize_weights(self):
        """
        Initialize the weights of the network
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save(self, filename=None):
        """
        Save the model into the filename
        :arg
            filename: file in which to save the model
        """
        filename = filename if filename is not None else self.__class__.__name__ + '.pt'
        torch.save(self.state_dict(), filename)

    def load_weights(self, file_path):
        """
        Load the model's weights saved into the filename
        :arg
            file_path: path file where model's weights are saved
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(file_path, map_location=device))
