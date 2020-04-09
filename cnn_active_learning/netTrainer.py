# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import warnings
import torch
import numpy as np
from dataManager import DataManager
from typing import Callable, Type
from tqdm import tqdm
import matplotlib.pyplot as plt
from selector import Selector, UncertaintySelector, MarginSamplingSelector


class NetTrainer(object):
    """
    Class used the train and test the given model in the parameters 
    """

    def __init__(self, model,
                 data_manager: DataManager,
                 selection_method: Selector, 
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 batch_size=1,
                 use_cuda=False):
        """
        Args:
            model: model to train
            trainset: dataset used to train the model
            testset: dataset used to test the model
            loss_fn: the loss function used
            optimizer_factory: A callable to create the optimizer. see optimizer function below for more details
            use_cuda: to Use the gpu to train the model
        """

        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        
        self.data_manager = data_manager
        self.selection_method = selection_method
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer_factory(self.model)
        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        self.metric_values = {}

    def train(self, num_epochs):
        """
        Train the model for num_epochs times
        Args:
            num_epochs: number times to train the model
        """
        print('Training...')

        # Initialize metrics container
        self.metric_values['train_loss'] = []
        self.metric_values['train_acc'] = []
        self.metric_values['val_loss'] = []
        self.metric_values['val_acc'] = []

        # Create pytorch's train data_loader
        train_loader = self.data_manager.get_train_loader()
        
        # train num_epochs times
        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0
            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, train_labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)

                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_labels)

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    """
                    Way it could be done manually

                    with torch.no_grad():
                        for param in self.model.parameters():
                            param -= learning_rate * param.grad
                    """
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    train_accuracies.append(self.accuracy(train_outputs, train_labels))

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

        print("Finished training.")


    def evaluate_on_validation_set(self, k:int):
        """
        function that evaluate the model on the validation set every epoch
        """
        print('Evaluate on validation set...')
        # switch to eval mode so that layers like batchnorm's layers nor dropout's layers
        # works in eval mode instead of training mode
        self.model.eval()

        # Get validation data
        val_loader = self.data_manager.get_validation_loader()
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []
        maxprobas = None
        selection = None

        with torch.no_grad() and tqdm(range(len(val_loader))) as t:
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)

                # forward pass
                val_outputs = self.model(val_inputs)
  
                # compute loss function
                loss = self.loss_fn(val_outputs, val_labels)
                validation_losses.append(loss.item())
                acc = self.accuracy(val_outputs, val_labels)
                validation_accuracies.append(acc)
                validation_loss += loss.item()
               
                # select indexes for next training
                selection = self.selection_method.select(val_outputs, j, k, selection)

                t.update()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))

        # switch back to train mode
        self.model.train()

        return self.selection_method.indexes(selection)

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of the model
        Args:
            outputs: outputs predicted by the model
            labels: real outputs of the data
        Returns:
            Accuracy of the model
        """
        predicted = outputs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set
        :returns;
            Accuracy of the model on the test set
        """
        print('Evaluate on test set...')
        test_loader = self.data_manager.get_test_loader()
        accuracies = 0
        with torch.no_grad() and tqdm(range(len(test_loader))) as t:
            for i, data in enumerate(test_loader, 0):
                test_inputs, test_labels = data[0].to(self.device), data[1].to(self.device)
                test_outputs = self.model(test_inputs)
                accuracies += self.accuracy(test_outputs, test_labels)
                t.update()

        percent_accuracy = round(100 * accuracies / len(test_loader), 3)
        print("Accuracy of the network on the test set: "+ str(percent_accuracy))
        return percent_accuracy


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f
