from dataExtractor import CIFAR10Extractor 
import torch 
import torch.nn as nn
from dataManager import DataManager
from netTrainer import optimizer_setup, NetTrainer
from networks.VggNet import VggNet
import matplotlib.pyplot  as plt
import numpy as np
import torchvision

data = CIFAR10Extractor()

num_learnings = 2
k=50
idx_labeled_samples = np.arange(k)

for num_learning in range(num_learnings):
    print('\nIndexes of samples for training:\n',idx_labeled_samples)
    dataManager = DataManager(data=data, \
        idx_labeled_samples=idx_labeled_samples, \
        batch_size=10)

    optimizer = optimizer_setup(torch.optim.SGD, lr=0.001, momentum=0.9)
    vgg = VggNet(num_classes=10)

    netTrainer = NetTrainer(model=vgg, \
            data_manager=dataManager, \
           loss_fn=nn.CrossEntropyLoss() , \
           optimizer_factory=optimizer)

    netTrainer.train(2)
    selected_samples_idx = netTrainer.evaluate_on_validation_set(k)
    add_to_train_idx = selected_samples_idx[1,:].astype(int)
    print('k indexes of lower confidence classification:\n',add_to_train_idx)
    idx_labeled_samples = np.concatenate((idx_labeled_samples, add_to_train_idx))

