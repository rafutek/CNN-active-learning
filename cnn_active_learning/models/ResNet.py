# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
import torch.nn.functional as F
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import ResidualBlock


class ResNet(CNNBaseModel):
    """
    Class that implements the ResNet 18 layers model.
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    """

    def __init__(self, num_classes=10, init_weights=True):
        """
        Builds ResNet-18 model.
        Args:
            num_classes(int): number of classes. default 200(tiny imagenet)
    
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(ResNet, self).__init__(num_classes, init_weights)

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_resnet18_layer(64, stride=1)
        self.layer2 = self._make_resnet18_layer(128, stride=2)
        self.layer3 = self._make_resnet18_layer(256, stride=2)
        self.layer4 = self._make_resnet18_layer(512, stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_resnet18_layer(self, out_channels, stride):
        """
        Building ResNet layer
        """
        strides = [stride] + [1]
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output, 2)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        output = self.softmax(output)
        return output
