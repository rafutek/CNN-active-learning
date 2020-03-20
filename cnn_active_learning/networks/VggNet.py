# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
import torch
from networks.CNNBaseModel import CNNBaseModel


class VggNet(CNNBaseModel):
    """
    Class that implements the vgg 16 layers model from
    "Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>

    """

    def __init__(self, num_classes=10, init_weights=True):
        """
        Builds VGG-16 model.
        Args:
            num_classes(int): number of classes. default 200(tiny imagenet)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(VggNet, self).__init__(num_classes, init_weights)

        self.conv_layers = self._make_vgg16_conv_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512 , 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        self.soft_classifier = nn.Sequential(
            nn.Linear(512 , 4096),
            nn.Linear(4096 , 1000),
            nn.Linear(1000 , num_classes),
            nn.Softmax(dim=1),
        )

    @staticmethod
    def _make_vgg16_conv_layers(batch_norm=True):
        """
        Build the convolutional part of the network  layer by layer by looping through the vgg_config
        list variable.
        Args:
            batch_norm: if true use the batchnormalization
        """
        vgg_config = [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP']
        layers = []
        in_channels = 3
        for x in vgg_config:
            if x == 'MP':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Reshape feature maps
        x = self.soft_classifier(x)
        return x
