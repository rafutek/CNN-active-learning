# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: ...
Other: Suggestions are welcome
"""

from dataExtractor import DataExtractor
import torch
from dataset import Dataset
from torch.utils.data import DataLoader


class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, data : DataExtractor,
                idx_labeled_samples):
        
        pool_samples, pool_labels = data.get_pool_data()
        test_samples, test_labels = data.get_test_data()

        if len(idx_labeled_samples) >= len(pool_samples):
            raise AttributeError("labeled samples length must be smaller than pool length")
       
        # Split pool in train and validation set
        train_samples, train_labels, val_samples, val_labels = \
                self.train_validation_split(pool_samples, pool_labels, idx_labeled_samples)

        train_set = Dataset(train_samples, train_labels)
        val_set = Dataset(val_samples, val_labels)
        test_set = Dataset(test_samples, test_labels)
        
        self.train_loader = DataLoader(train_set, shuffle=True)
        self.validation_loader = DataLoader(val_set, shuffle=True)
        self.test_loader = DataLoader(test_set, shuffle=True)

    def train_validation_split(self, pool_samples, pool_labels, idx_labeled_samples):
        train_samples = pool_samples[idx_labeled_samples]
        train_labels = pool_labels[idx_labeled_samples]
        
        minusOne = torch.zeros(len(pool_samples)).fill_(-1).type(dtype=torch.long)
        idx_samples = minusOne.clone()
        idx_samples[idx_labeled_samples] = idx_labeled_samples.clone().detach()
        
        idx = torch.arange(len(pool_samples))
        idx_val = torch.where(idx != idx_samples,idx, minusOne)
        idx_val = idx_val[idx_val != -1]
        
        val_samples = pool_samples[idx_val]
        val_labels = pool_labels[idx_val]

        return train_samples, train_labels, val_samples, val_labels

    def get_train_loader(self):
        return self.train_loader

    def get_validation_loader(self):
        return self.validation_loader

    def get_test_loader(self):
        return self.test_loader

