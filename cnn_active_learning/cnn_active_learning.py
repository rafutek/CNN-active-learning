from dataExtractor import CIFAR10Extractor 
import torch 
import torch.nn as nn
from dataManager import DataManager
from netTrainer import optimizer_setup, NetTrainer
from networks.VggNet import VggNet
import matplotlib.pyplot  as plt
import numpy as np
import torchvision

num_learnings = 2
k=50
num_epochs = 2
batch_size = 10

data = CIFAR10Extractor()

idx_labeled_samples = np.arange(k)
dataManager = DataManager(data=data, \
        idx_labeled_samples=idx_labeled_samples, \
        batch_size=batch_size)

num_classes = len(data.get_label_names())
idx_labeled_samples = np.arange(k)

accuracies = []

for num_learning in range(num_learnings):
    print('\nIndexes of samples for training:\n',idx_labeled_samples)
    dataManager = DataManager(data=data, \
        idx_labeled_samples=idx_labeled_samples, \
        batch_size=batch_size)

    optimizer = optimizer_setup(torch.optim.SGD, lr=0.001, momentum=0.9)
    vgg = VggNet(num_classes=num_classes)

    netTrainer = NetTrainer(model=vgg, \
            data_manager=dataManager, \
           loss_fn=nn.CrossEntropyLoss() , \
           optimizer_factory=optimizer)

    netTrainer.train(num_epochs)
    selected_samples_idx = netTrainer.evaluate_on_validation_set(k)
    add_to_train_idx = selected_samples_idx
    print('k indexes of lower confidence classification:\n',add_to_train_idx)
    idx_labeled_samples = np.concatenate((idx_labeled_samples, add_to_train_idx))

    accuracy = netTrainer.evaluate_on_test_set()
    accuracies.append(accuracy)
    print(accuracies)

