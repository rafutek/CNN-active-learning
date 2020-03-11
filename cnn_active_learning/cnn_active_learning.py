from dataset import CIFAR10
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

CIFAR10.download()
data,labels,filenames = CIFAR10.unpickle('./data/cifar-10-batches-py/data_batch_3')
# print(data.shape, len(labels), len(filenames))
print(labels[0], filenames[0])


