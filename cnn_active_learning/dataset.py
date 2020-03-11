import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import subprocess
import numpy as np

# class model
class MyDataset(object):

    def __init__(self):
        pass

    def download(self):
        pass
    
    def get_data(self):
        pass

class CIFAR10(MyDataset):

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='latin1')
        return dic

    # Call a script to download CIFAR10 dataset
    @staticmethod
    def download():
        subprocess.call(["./dl-CIFAR10.sh"])
    
    @staticmethod 
    def get_data():

        data_dir = "./data/cifar-10-batches-py/"
        filenames = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
        data = None
        labels = None
        for filename in filenames:
            filepath = data_dir + filename
            data_dic = CIFAR10.unpickle(filepath)
            if data is None:
                #data = np.empty(data_dic['data'].shape, len(filenames))
                data = data_dic['data']
                labels = data_dic['labels']
                print(len(data))
                print(len(labels))
            else:

                data = np.append(data, data_dic['data'], axis=0)
                labels = np.append(labels, data_dic['labels'])
                print(len(data))

        label_dic = CIFAR10.unpickle('./data/cifar-10-batches-py/batches.meta')
        label_names = label_dic['label_names']
        return data, labels, label_names

