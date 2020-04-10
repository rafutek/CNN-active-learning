import numpy as np


class Selector(object):
    """
    Base of a selection method class
    """
    def __init__(self, val_idx, outputs):
        self.val_idx = val_idx
        self.outputs = outputs

    def compute_selection_parameters(self):
        pass
    def order_val_idx(self):
        pass
    def get_k_idx(self, k):
        """
        Get the first k indexes previously ordered
        Args:
            k: number of indexes wanted
        Returns:
            k indexes indicating the samples to use
            for the next training
        """
        raise NotImplemented

    def select(self, k):
        self.compute_selection_parameters()
        self.order_val_idx()
        return self.get_k_idx(k)


class RandomSelector(Selector):
    def order_val_idx(self):
        """
        Function that shuffle the validation indexes
        so that the first k indexes will be random
        """
        self.random_idx = self.val_idx
        np.random.shuffle(self.random_idx)

    def get_k_idx(self,k):
        return self.random_idx[:k].astype(int)


class UncertaintySelector(Selector):
    @staticmethod
    def select(network_output, num_loop, k, selection):
        batch_size = len(network_output)
        sample_numbers = np.arange(batch_size) + k + batch_size*num_loop # samples index in the pool
        maxprobability, _ = network_output.max(dim=1)
        batch_maxprobas = np.array([maxprobability.tolist(), sample_numbers.tolist()])
        if selection is None:
            selection = batch_maxprobas
        else:
            selection = np.concatenate((selection, batch_maxprobas), axis=1)

        if selection.shape[1] > k:
            sort_idx = selection[0,:].argsort()
            selection =  selection[:,sort_idx] # sort maximums
            selection = selection[:,:k] # keep k minimal maximums
        return selection

    @staticmethod
    def indexes(selection):
        return selection[1,:].astype(int)


class MarginSamplingSelector(Selector):
    """
    Class to select validation samples for next training
    with the margin sampling selection method
    """
    def compute_selection_parameters(self):
        """
        Function that computes the parameter
        of each sample based on margin sampling
        """
        # Sort the class probabilities of each sample
        sorted_probas = np.sort(self.outputs)

        # Keep the first and second 
        # maximum probability of each sample
        maxproba = sorted_probas[:,-1] 
        second_maxproba = sorted_probas[:,-2] 

        # Subtract the first with the second
        # max proba of each sample
        self.parameters = maxproba - second_maxproba

    def order_val_idx(self):
        """
        Function that order validation sample indexes
        based on the parameter of each sample
        """
        # Create an array containing index and parameter of each sample
        param_with_idx = np.array([self.val_idx, self.parameters])
        
        # Get the indexes of sorted parameters
        # and then the associated sample indexes
        sort_idx = param_with_idx[1,:].argsort()
        self.sample_idx =  param_with_idx[0,sort_idx]

    def get_k_idx(self,k):
        return self.sample_idx[:k].astype(int)
