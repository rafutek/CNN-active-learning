import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import subprocess

# class model
class MyDataset(object):

    def __init__(self):
        pass

    def download(self):
        pass


class CIFAR10(MyDataset):

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dico = pickle.load(fo, encoding='latin1')
        data = dico.get('data')
        labels = dico.get('labels') #narray
        filenames = dico.get('filenames')
        return data,labels,filenames

    # Call a script to download CIFAR10 dataset
    @staticmethod
    def download():
        subprocess.call(["./dl-CIFAR10.sh"])
    

