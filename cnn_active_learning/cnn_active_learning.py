from dataset import CIFAR10
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

#CIFAR10.download()
dataset = CIFAR10()

data,labels,label_names = dataset.get_data()



print(data.shape, len(labels), len(label_names))
#print(labels[0], filenames[0])


