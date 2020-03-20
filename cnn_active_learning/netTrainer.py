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
# from utils import mean_dice, convert_mask_to_rgb_image
import matplotlib.pyplot as plt


class NetTrainer(object):
    """
    Class used the train and test the given model in the parameters 
    """

    def __init__(self, model,
                 data_manager: DataManager,
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
                
                maxprobability, _ = val_outputs.max(dim=1)
                batch_size = len(maxprobability)
                image_idx = torch.arange(batch_size)
                # print(image_idx)
                image_idx += k
                # print(image_idx)
                image_idx += batch_size*j # index in the pool
                # print(image_idx, '\n')
                batch_maxprobas = np.array([maxprobability.tolist(), image_idx.tolist()])
                if maxprobas is None:
                    maxprobas = batch_maxprobas
                else:
                    maxprobas = np.concatenate((maxprobas, batch_maxprobas), axis=1)

                if maxprobas.shape[1] > k:
                    sort_idx = maxprobas[0,:].argsort()
                    maxprobas =  maxprobas[:,sort_idx] # sort maximums
                    maxprobas = maxprobas[:,:k] # keep k minimal maximums
                t.update()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))

        # switch back to train mode
        self.model.train()

        return maxprobas

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
        test_loader = self.data_manager.get_test_loader()
        accuracies = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, test_labels = data[0].to(self.device), data[1].to(self.device)
                test_outputs = self.model(test_inputs)
                accuracies += self.accuracy(test_outputs, test_labels)
        print("Accuracy of the network on the test set: {:05.3f} %".format(100 * accuracies / len(test_loader)))

    def plot_metrics(self):
        """
        Function that plots train and validation losses and accuracies after training phase
        """
        epochs = range(1, len(self.metric_values['train_loss']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(epochs, self.metric_values['train_loss'], '-o', label='Training loss')
        ax1.plot(epochs, self.metric_values['val_loss'], '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(epochs, self.metric_values['train_acc'], '-o', label='Training accuracy')
        ax2.plot(epochs, self.metric_values['val_acc'], '-o', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        f.savefig('fig1.png')
        plt.show()

    def plot_image_mask_prediction(self):
        """
        Function that plots an image its corresponding ground truth and the predicted mask
        """
        # pick randomly an image and its corresponding gt into test_set
        img, gt = self.data_manager.get_random_sample_from_test_set()
        # convert the ground truth to a rgb image
        gt = convert_mask_to_rgb_image(gt)
        # use the model to predict the segmented image
        # Since the model expect a 4D we need to add the batch dim in order to get the 4D
        img = np.expand_dims(img, axis=0)
        # convert image to Tensor
        img = torch.from_numpy(img)
        img = img.to(self.device)
        prediction = self.model(img)
        # delete the batch dimension
        prediction = np.squeeze(prediction)
        # convert prediction to numpy array
        # take into account if model is trained on cpu or gpu
        if self.use_cuda and torch.cuda.is_available():
            prediction = prediction.detach().cpu().numpy()
        else:
            prediction = prediction.detach().numpy()
        # from one_hot vector to categorical
        prediction = np.argmax(prediction, axis=0)
        # convert the predicted mask to rgb image
        prediction = convert_mask_to_rgb_image(prediction)
        # remove the batch dim and the channel dim of img
        img = np.squeeze(np.squeeze(img))
        # convert img to a numpy array
        if self.use_cuda and torch.cuda.is_available():
            img = img.cpu().numpy()
        else:
            img = img.numpy()

        # plot the image
        f = plt.figure(figsize=(10, 10))
        ax1 = f.add_subplot(221)
        ax1.imshow(img)
        ax1.set_title('image')
        ax1.axis('off')

        # plot the ground truth
        ax2 = f.add_subplot(222)
        ax2.imshow(gt.astype('uint8'))
        ax2.axis('off')
        ax2.set_title('ground truth')

        # plot the image
        ax3 = f.add_subplot(223)
        ax3.imshow(img)
        ax3.set_title('image')
        ax3.axis('off')

        # plot the predicted mask
        ax4 = f.add_subplot(224)
        ax4.imshow(prediction.astype('uint8'))
        ax4.set_title('predicted segmentation')
        ax4.axis('off')

        # Save as a png image
        f.savefig('fig2.png')
        # show image
        plt.show()


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
