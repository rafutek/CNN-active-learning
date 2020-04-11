import subprocess, pickle
import numpy as np
import matplotlib.pyplot  as plt


class DataExtractor(object):
    def __init__(self):
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
        return self.pool_samples, self.pool_labels
    def get_test_data(self):
        return self.test_samples, self.test_labels
    def get_label_names(self):
        return self.label_names


class CIFAR10Extractor(DataExtractor):
    data_dir = './data/cifar-10-batches-py/'
    
    def download(self):
        subprocess.call(['./dl-CIFAR10.sh'])

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='latin1')
        return dic
   
    def extract_pool_data(self):
        filenames = ['data_batch_1'] # reduce data size for dev (faster)
        # filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
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
        
        pool_samples = pool_samples[0:500]
        pool_labels = pool_labels[0:500]
        pool_samples = np.vstack(pool_samples).reshape(-1, 3, 32, 32)
        return pool_samples, pool_labels
    
    @staticmethod
    def show_image(images, index):
        images = images.transpose((0, 2, 3, 1))
        img = images[index]
        plt.imshow(img)
        plt.show()

    def extract_test_samples(self):
        filepath = self.data_dir + 'test_batch'
        test_dic = self.unpickle(filepath)
        test_samples = test_dic['data']
        test_labels = test_dic['labels']
        test_samples = test_samples [0:500]
        test_labels = test_labels[0:500]
        test_samples = np.vstack(test_samples).reshape(-1, 3, 32, 32)
        return test_samples, test_labels

    def extract_label_names(self):
        filepath = self.data_dir + 'batches.meta'
        label_dic = self.unpickle(filepath)
        label_names = label_dic['label_names']
        return label_names




class CIFAR100Extractor(DataExtractor):
    data_dir = './data/cifar-100-python/'

    def download(self):
        subprocess.call(['./dl-CIFAR100.sh'])

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='latin1')
        return dic

    def extract_pool_data(self):
        filenames = ['train']
        init = False
        for filename in filenames:
            filepath = self.data_dir + filename
            pool_dic = self.unpickle(filepath)
            print(pool_dic.keys())
            data = pool_dic['data']
            labels = pool_dic['coarse_labels'] # ERR: il faut voir quels labels on utilise
            if not init:
                init = True
                pool_samples = np.array(data)
                pool_labels = np.array(labels)
            else:
                pool_samples = np.append(pool_samples, data, axis=0)
                pool_labels = np.append(pool_labels, labels)

        pool_samples = np.vstack(pool_samples).reshape(-1, 3, 32, 32)
        return pool_samples, pool_labels

    @staticmethod
    def show_image(images, index):
        images = images.transpose((0, 2, 3, 1))
        img = images[index]
        plt.imshow(img)
        plt.show()

    def extract_test_samples(self):
        filepath = self.data_dir + 'test'
        test_dic = self.unpickle(filepath)
        test_samples = test_dic['data']
        test_labels = test_dic['coarse_labels']
        test_samples = np.vstack(test_samples).reshape(-1, 3, 32, 32)
        return test_samples, test_labels

    def extract_label_names(self):
        filepath = self.data_dir + 'meta'
        label_dic = self.unpickle(filepath)
        label_names = label_dic['coarse_label_names']
        return label_names
