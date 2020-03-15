import subprocess, pickle
import numpy as np
import matplotlib.pyplot  as plt


class DataExtractor(object):
    def __init__(self):
        self.download()
        self.pool_samples, self.pool_labels = self.extract_pool_data()
        self.test_data, self.test_labels = self.extract_test_data()       
        self.label_names = self.extract_label_names()

    def download(self):
        raise NotImplementedError
    def extract_pool_data(self):
        raise NotImplementedError
    def extract_test_data(self):
        raise NotImplementedError
    def extract_label_names(self):
        raise NotImplementedError

    def get_pool_data(self):
        return self.pool_samples, self.pool_labels
    def get_test_data(self):
        return self.test_data, self.test_labels
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
        filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
        init = False
        for filename in filenames:
            filepath = self.data_dir + filename
            pool_dic = self.unpickle(filepath)
            data = pool_dic['data']
            labels = pool_dic['labels']
            if not init:
                init = True
                pool_samples = data
                pool_labels = labels
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

    def extract_test_data(self):
        filepath = self.data_dir + 'test_batch'
        test_dic = self.unpickle(filepath)
        test_data = test_dic['data']
        test_labels = test_dic['labels']
        return test_data, test_labels

    def extract_label_names(self):
        label_dic = self.unpickle('./data/cifar-10-batches-py/batches.meta')
        label_names = label_dic['label_names']
        return label_names

