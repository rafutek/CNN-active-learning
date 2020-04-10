# -*- coding:utf-8 -*-

import numpy as np
from dataExtractor import DataExtractor
from dataset import Dataset
from torch.utils.data import DataLoader


class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, data : DataExtractor, 
            idx_labeled_samples: np.array,
            batch_size: int = 20):
        
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
        
        self.train_loader = DataLoader(train_set, batch_size)
        self.validation_loader = DataLoader(val_set, batch_size)
        self.test_loader = DataLoader(test_set, batch_size)

    def train_validation_split(self, pool_samples, pool_labels, idx_labeled_samples):
        pool_length = len(pool_samples)

        # Set the train samples and labels
        mask_train = np.zeros(pool_length ,dtype=bool)
        mask_train[idx_labeled_samples] = True
        train_samples = pool_samples[mask_train]
        train_labels = pool_labels[mask_train]

        # Set the validation samples and labels
        mask_val = np.ones(pool_length,dtype=bool)
        mask_val[idx_labeled_samples] = False
        val_samples = pool_samples[mask_val]
        val_labels = pool_labels[mask_val]

        # Set the train and validation samples indexes
        pool_idx = np.arange(pool_length)
        self.train_idx = pool_idx[mask_train]
        self.val_idx = pool_idx[mask_val]

        return train_samples, train_labels, val_samples, val_labels

    def get_train_loader(self):
        return self.train_loader

    def get_validation_loader(self):
        return self.validation_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_idx(self):
        return self.train_idx

    def get_val_idx(self):
        return self.val_idx
