from networks.VggNet import VggNet
from dataExtractor import CIFAR10Extractor 
from dataManager import DataManager
from netTrainer import optimizer_setup, NetTrainer
import numpy as np
from torch.optim import SGD 
import torch.nn as nn

def active_learning(network, dataset, method, k, num_trainings,
                    batch_size, num_epochs, learning_rate):
    print("Starting active learning with"
            "\n\tmodel: "+network+
            "\n\tdataset: "+dataset+
            "\n\tselection method: "+method+
            "\n\tk: "+str(k)+
            "\n\tnum trainings: "+str(num_trainings)+
            "\n\tbatch size: "+str(batch_size)+
            "\n\tnum epochs: "+str(num_epochs)+
            "\n\tlearning rate: "+str(learning_rate)
    )

    model = getModel(network)
    data = getData(dataset)
    selection_method = getSelectionMethod(method)
    k = int(k)

    idx_labeled_samples = np.arange(k)
    dataManager = DataManager(data=data, \
            idx_labeled_samples=idx_labeled_samples, \
            batch_size=batch_size)
    
    num_classes = len(data.get_label_names())
    
    accuracies = []
    
    for num_training in range(num_trainings):
        print("\nActive learning loop "+str(num_training+1)+"/"+str(num_trainings))
        # print('\nIndexes of samples for training:\n',idx_labeled_samples)
        dataManager = DataManager(data=data, \
            idx_labeled_samples=idx_labeled_samples, \
            batch_size=batch_size)
    
        optimizer = optimizer_setup(SGD, lr=0.001, momentum=0.9)
        vgg = VggNet(num_classes=num_classes)
    
        netTrainer = NetTrainer(model=vgg, \
                data_manager=dataManager, \
               loss_fn=nn.CrossEntropyLoss() , \
               optimizer_factory=optimizer)
    
        netTrainer.train(num_epochs)
        selected_samples_idx = netTrainer.evaluate_on_validation_set(k)
        add_to_train_idx = selected_samples_idx
        print('k indexes of lower confidence classification:\n',add_to_train_idx)
        idx_labeled_samples = np.concatenate((idx_labeled_samples, add_to_train_idx))
    
        accuracy = netTrainer.evaluate_on_test_set()
        accuracies.append(accuracy)
    
    return accuracies

def getModel(model:str):
    if model == "VggNet":
        return VggNet
    elif model == "ResNeXt":
        return None

def getData(dataset):
    if dataset == "cifar10":
        return CIFAR10Extractor()
    elif dataset == "cifar100":
        return CIFAR100Extractor()

def getSelectionMethod(method):
    if method == "random":
        return None
    elif method == "uncertainty_sampling":
        return None
