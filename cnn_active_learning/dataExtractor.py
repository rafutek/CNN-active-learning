import subprocess, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
# import torchaudio
# import pandas as pd

class DataExtractor(object):
    """
    Base class for data downloading and extraction in pool and test set
    """
    def __init__(self):
        """
        Constructor that download the dataset
        and extract the pool set, the test set and the label names
        """
        self.download()
        self.pool_samples, self.pool_labels = self.extract_pool_data()
        self.test_samples, self.test_labels = self.extract_test_samples()       
        self.label_names = self.extract_label_names()

    def download(self):
        raise NotImplementedError
    def extract_pool_data(self):
        raise NotImplementedError
    def extract_test_samples(self):
        raise NotImplementedError
    def extract_label_names(self):
        raise NotImplementedError

    def get_pool_data(self):
        """
        Function to get the pool data
        Returns:
            The pool samples and associated labels
        """
        return self.pool_samples, self.pool_labels

    def get_test_data(self):
        """
        Function to get the test data
        Returns:
            The test samples and associated labels
        """
        return self.test_samples, self.test_labels

    def get_label_names(self):
        """
        Function to get the label names
        Returns:
            The names corresponding to each label
        """
        return self.label_names

    def unpickle(self, f):
        """
        Function to load a pickle file in a dictionary
        Args:
            f: name of the file to unpickle
        Returns:
            The dictionary containing the file data
        """
        with open(f, 'rb') as fo:
            dic = pickle.load(fo, encoding='latin1')
        return dic


class CIFAR10Extractor(DataExtractor):
    """
    Class to manage CIFAR10 downloading and
    extraction in pool and test sets
    """

    data_dir = './data/cifar-10-batches-py/'
    
    def download(self):
        """
        Function that call the script to download
        the CIFAR10 dataset and unzip it
        """
        subprocess.call(['./scripts/dl-CIFAR10.sh'])
   
    def extract_pool_data(self):
        """
        Function to load the CIFAR10 data from files
        in order to create the pool set
        Returns:
            The pool samples and corresponding labels
        """
        filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
        init = False
        for filename in filenames:
            filepath = self.data_dir + filename
            pool_dic = self.unpickle(filepath)
            data = pool_dic['data']
            labels = pool_dic['labels']
            if not init:
                init = True
                pool_samples = np.array(data)
                pool_labels = np.array(labels)
            else:
                pool_samples = np.append(pool_samples, data, axis=0)
                pool_labels = np.append(pool_labels, labels)
        
        pool_samples = pool_samples[0:1000]
        pool_labels = pool_labels[0:1000]
        pool_samples = np.vstack(pool_samples).reshape(-1, 3, 32, 32)
        return pool_samples, pool_labels

    def extract_test_samples(self):
        """
        Function to load the CIFAR10 data from files
        in order to create the test set
        Returns:
            The test samples and corresponding labels
        """
        filepath = self.data_dir + 'test_batch'
        test_dic = self.unpickle(filepath)
        test_samples = test_dic['data']
        test_labels = test_dic['labels']
        test_samples = np.vstack(test_samples).reshape(-1, 3, 32, 32)
        return test_samples, test_labels

    def extract_label_names(self):
        """
        Function to load the CIFAR10 label names
        Returns:
            The label names
        """
        filepath = self.data_dir + 'batches.meta'
        label_dic = self.unpickle(filepath)
        label_names = label_dic['label_names']
        return label_names




class CIFAR100Extractor(DataExtractor):
    """
    Class to manage CIFAR100 downloading and
    extraction in pool and test sets
    """

    data_dir = './data/cifar-100-python/'

    def download(self):
        """
        Function that call the script to download
        the CIFAR100 dataset and unzip it
        """
        subprocess.call(['./scripts/dl-CIFAR100.sh'])

    def extract_pool_data(self):
        """
        Function to load the CIFAR100 data from files
        in order to create the pool set
        Returns:
            The pool samples and corresponding labels
        """
        filenames = ['train']
        init = False
        for filename in filenames:
            filepath = self.data_dir + filename
            pool_dic = self.unpickle(filepath)
            data = pool_dic['data']
            labels = pool_dic['coarse_labels']
            if not init:
                init = True
                pool_samples = np.array(data)
                pool_labels = np.array(labels)
            else:
                pool_samples = np.append(pool_samples, data, axis=0)
                pool_labels = np.append(pool_labels, labels)

        pool_samples = np.vstack(pool_samples).reshape(-1, 3, 32, 32)
        return pool_samples, pool_labels

    def extract_test_samples(self):
        """
        Function to load the CIFAR100 data from files
        in order to create the test set
        Returns:
            The test samples and corresponding labels
        """
        filepath = self.data_dir + 'test'
        test_dic = self.unpickle(filepath)
        test_samples = test_dic['data']
        test_labels = test_dic['coarse_labels']
        test_samples = np.vstack(test_samples).reshape(-1, 3, 32, 32)
        return test_samples, test_labels

    def extract_label_names(self):
        """
        Function to load the CIFAR10 label names
        Returns:
            The label names
        """
        filepath = self.data_dir + 'meta'
        label_dic = self.unpickle(filepath)
        label_names = label_dic['coarse_label_names']
        return label_names



class UrbanSoundExtractor(DataExtractor):
    """
    Class to manage CIFAR100 downloading and
    extraction in pool and test sets
    """
    
    data_dir = './data/UrbanSound/'

    def download(self):
        """
        Function that call the script to download
        the UrbanSound dataset and unzip it
        """
        subprocess.call(['./scripts/dl-audioset.sh'])

    def extract_pool_data(self):
        raise NotImplementedError

    def extract_test_samples(self):
        raise NotImplementedError

    def extract_label_names(self):
        raise NotImplementedError




# TODO: Adapt to a data extractor
# class UrbanSoundDataset(Dataset):
#     # rapper for the UrbanSound8K dataset
#     # Argument List
#     #  path to the UrbanSound8K csv file
#     #  path to the UrbanSound8K audio files
#     #  list of folders to use in the dataset

#     def __init__(self, csv_path, file_path, folderList):
#         csvData = pd.read_csv(csv_path)
#         # initialize lists to hold file names, labels, and folder numbers
#         self.file_names = []
#         self.labels = []
#         self.folders = []
#         # loop through the csv entries and only add entries from folders in the folder list
#         for i in range(0, len(csvData)):
#             if csvData.iloc[i, 5] in folderList:
#                 self.file_names.append(csvData.iloc[i, 0])
#                 self.labels.append(csvData.iloc[i, 6])
#                 self.folders.append(csvData.iloc[i, 5])

#         self.file_path = file_path
#         self.mixer = torchaudio.transforms.DownmixMono()  # UrbanSound8K uses two channels, this will convert them to one
#         self.folderList = folderList

#     def __getitem__(self, index):
#         # format the file path and load the file
#         path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
#         sound = torchaudio.load(path, out=None, normalization=True)
#         # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
#         soundData = self.mixer(sound[0])
#         # downsample the audio to ~8kHz
#         tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
#         if soundData.numel() < 160000:
#             tempData[:soundData.numel()] = soundData[:]
#         else:
#             tempData[:] = soundData[:160000]

#         soundData = tempData
#         soundFormatted = torch.zeros([32000, 1])
#         soundFormatted[:32000] = soundData[::5]  # take every fifth sample of soundData
#         soundFormatted = soundFormatted.permute(1, 0)
#         return soundFormatted, self.labels[index]

#     def __len__(self):
#         return len(self.file_names)


# csv_path = './Data/UrbanSound8K.csv'
# file_path = './UrbanSound8K/audio/'

# train_set = UrbanSoundDataset(csv_path, file_path, range(1, 10))
# test_set = UrbanSoundDataset(csv_path, file_path, [10])
# print("Train set size: " + str(len(train_set)))
# print("Test set size: " + str(len(test_set)))

# kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)
