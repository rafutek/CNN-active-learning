import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import subprocess
import numpy as np
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self):
        self.download()
        self.train_data, self.labels, self.label_names = self.get_train_data()
        self.test_data, self.test_labels = self.get_test_data()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index], self.labels[index], self.label_names[self.labels[index]]

    def download(self):
        pass

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass



class CIFAR10(ModelDataset):
    data_dir = './data/cifar-10-batches-py/'
    
    def download(self):
        subprocess.call(['./dl-CIFAR10.sh'])

    def get_train_data(self):

        filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
        data = None
        for filename in filenames:
            filepath = self.data_dir + filename
            data_dic = self.unpickle(filepath)
            if data is None:
                data = data_dic['data']
                labels = data_dic['labels']
            else:
                data = np.append(data, data_dic['data'], axis=0)
                labels = np.append(labels, data_dic['labels'])

        label_dic = self.unpickle('./data/cifar-10-batches-py/batches.meta')
        label_names = label_dic['label_names']
        return data, labels, label_names

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='latin1')
        return dic
    
    def get_test_data(self):
        filepath = self.data_dir + 'test_batch'
        test_dic = self.unpickle(filepath)
        test_data = test_dic['data']
        test_labels = test_dic['labels']
        return test_data, test_labels
