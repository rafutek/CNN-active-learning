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
            train_idx: np.array,
            batch_size: int = 20):

        """
        Constructor that split pool data in train and validation sets
        depending on training sample indexes, and create data loaders
        for train, validation and test sets
        Args:
            data: object containing the pool and test data
            train_idx: array indicating the indexes in the pool 
                    of the samples to train
            batch_size: number of samples in a batch
        """
        
        pool_samples, pool_labels = data.get_pool_data()
        test_samples, test_labels = data.get_test_data()

        if len(train_idx) >= len(pool_samples):
            raise AttributeError("labeled samples length must be smaller than pool length")
       
        # Split pool in train and validation set
        train_samples, train_labels, val_samples, val_labels = \
                self.train_validation_split(pool_samples, pool_labels, train_idx)

        train_set = Dataset(train_samples, train_labels)
        val_set = Dataset(val_samples, val_labels)
        test_set = Dataset(test_samples, test_labels)
        
        self.train_loader = DataLoader(train_set, batch_size)
        self.validation_loader = DataLoader(val_set, batch_size)
        self.test_loader = DataLoader(test_set, batch_size)

    def train_validation_split(self, pool_samples, pool_labels, train_idx):
        """
        Function that split pool set in train and validation set
        Args:
            pool_samples: array containing the pool samples
            pool_labels: array containing the pool labels
            train_idx: array containing the pool indexes
                    of the samples to train
        Returns:
            train_samples: array containing the train samples
            train_labels: array containing the train labels
            val_samples: array containing the validation samples
            val_labels: array containing the validation labels
        """
        pool_length = len(pool_samples)

        # Set the train samples and labels
        mask_train = np.zeros(pool_length ,dtype=bool)
        mask_train[train_idx] = True
        train_samples = pool_samples[mask_train]
        train_labels = pool_labels[mask_train]

        # Set the validation samples and labels
        mask_val = np.ones(pool_length,dtype=bool)
        mask_val[train_idx] = False
        val_samples = pool_samples[mask_val]
        val_labels = pool_labels[mask_val]

        # Set the train and validation samples indexes
        pool_idx = np.arange(pool_length)
        self.train_idx = pool_idx[mask_train]
        self.val_idx = pool_idx[mask_val]

        return train_samples, train_labels, val_samples, val_labels

    def get_train_loader(self):
        """
        Function to get the train loader
        Returns:
            A DataLoader object for training set
        """
        return self.train_loader

    def get_validation_loader(self):
        """
        Function to get the validation loader
        Returns:
            A DataLoader object for validation set
        """
        return self.validation_loader

    def get_test_loader(self):
        """
        Function to get the test loader
        Returns:
            A DataLoader object for test set
        """
        return self.test_loader

    def get_train_idx(self):
        """
        Function to get the pool indexes of the training samples
        Returns:
            The pool indexes of the training samples
        """
        return self.train_idx

    def get_val_idx(self):
        """
        Function to get the pool indexes of the validation samples
        Returns:
            The pool indexes of the validation samples
        """
        return self.val_idx
