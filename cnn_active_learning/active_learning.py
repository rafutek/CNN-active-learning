from models.VggNet import VggNet
from models.ResNet import ResNet
from models.AlexNet import AlexNet
from dataExtractor import CIFAR10Extractor, CIFAR100Extractor 
from dataManager import DataManager
from selector import RandomSelector, UncertaintySelector, MarginSamplingSelector
from netTrainer import optimizer_setup, NetTrainer
import numpy as np
from torch.optim import SGD 
import torch.nn as nn

def active_learning(network:str, dataset:str, method:str, k:str, num_trainings:int,
                    batch_size:int, num_epochs:int, learning_rate:float, use_cuda:bool):
    """
    Function that execute an active learning
    depending on several arguments
    Args:
        network: the model that will be trained and tested
        dataset: the data used for the active learning
        method: method used to select the k samples to add
                to the training set at each active learning loop
        k: number of samples to add
        num_trainings: number of active learning loops
        batch_size: number of samples in a batch
        num_epochs: number of loops during one training
        learning_rate: learning rate of the optimizer
        use_cuda: boolean to use the gpu for training
    Returns:
        The list of accuracies of each test phase
    """

    print("Starting active learning with"
            "\n\tmodel: "+network+
            "\n\tdataset: "+dataset+
            "\n\tselection method: "+method+
            "\n\tk: "+k+
            "\n\tnum trainings: "+str(num_trainings)+
            "\n\tbatch size: "+str(batch_size)+
            "\n\tnum epochs: "+str(num_epochs)+
            "\n\tlearning rate: "+str(learning_rate)+
            "\n\tuse cuda: "+str(use_cuda)
    )

    model = getModel(network)
    data = getData(dataset)
    selection_method = getSelectionMethod(method)
    k = int(k)

    if len(data.get_pool_data()[0]) < k * num_trainings:
        raise ValueError("'k' or 'num-trainings' is too big, "
                        "the program will not be able to extract the training "
                        "samples from the pool at some point")

    # Set the optimizer factory function
    optimizer = optimizer_setup(SGD, lr=learning_rate, momentum=0.9)

    # Create the network depending on the number of classes
    model = model(num_classes=len(data.get_label_names()))

    # First index samples to train
    idx_labeled_samples = np.arange(k)    
    
    # List that will contain the test accuracy of each training
    accuracies = []
    
    for num_training in range(num_trainings):
        print("\nActive learning loop "+str(num_training+1)+"/"+str(num_trainings))

        # Set data loaders depending on training samples
        dataManager = DataManager(data=data, \
            idx_labeled_samples=idx_labeled_samples, \
            batch_size=batch_size)

        # Set the network trainer and launch the training
        netTrainer = NetTrainer(model=model, \
                                data_manager=dataManager, \
                                selection_method=selection_method, \
                                loss_fn=nn.CrossEntropyLoss() , \
                                optimizer_factory=optimizer, \
                                use_cuda=use_cuda)
        netTrainer.train(num_epochs)

        # Select k samples depending on the selection method
        # and add them to the training samples
        add_to_train_idx = netTrainer.evaluate_on_validation_set(k)
        idx_labeled_samples = np.concatenate((idx_labeled_samples, add_to_train_idx))
        print("Added the selected samples to the new training set")

        # Compute the accuracy on the test set and save it
        accuracy = netTrainer.evaluate_on_test_set()
        accuracies.append(accuracy)
    
    return accuracies

def getModel(model:str):
    """
    Function to get the model object
    Args:
        model: string corresponding to the model object
    Returns:
        The model object
    """
    if model == "VggNet":
        return VggNet
    elif model == "ResNet":
        return ResNet
    elif model == "AlexNet":
        return AlexNet

def getData(dataset:str):
    """
    Function that download the wanted dataset
    if not already done and return the data object
    Args:
        dataset: string corresponding to the data object
    Returns:
        The data object initialized
    """
    if dataset == "cifar10":
        return CIFAR10Extractor()
    elif dataset == "cifar100":
        return CIFAR100Extractor()
    elif dataset == "audioset":
        return None

def getSelectionMethod(method):
    """
    Function to get the wanted selection method
    Args:
        method: string corresponding to the 
                selection method object
    Returns:
        The selection method object
    """
    if method == "random":
        return RandomSelector
    elif method == "uncertainty_sampling":
        return UncertaintySelector
    elif method == "margin_sampling":
        return MarginSamplingSelector
    elif method == "entropy_sampling":
        return EntropySamplingSelector
