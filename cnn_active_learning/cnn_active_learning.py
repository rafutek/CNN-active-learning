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
pool_samples, pool_labels = data.get_pool_data()
print(pool_samples.shape)


k=1000
idx_labeled_samples = torch.randint(size=(k,),high=len(pool_samples))

dataManager = DataManager(data, idx_labeled_samples)


optimizer = optimizer_setup(torch.optim.SGD, lr=0.001, momentum=0.9)
vgg = VggNet(num_classes=10)

netTrainer = NetTrainer(model=vgg, data_manager=dataManager, loss_fn=nn.CrossEntropyLoss() , optimizer_factory=optimizer)

netTrainer.train(10)
