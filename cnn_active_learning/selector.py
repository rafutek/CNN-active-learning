import numpy as np
from scipy.stats import entropy
class Selector(object):
    """
    Base of a selection method classes
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
        raise NotImplementedError("get_k_idx method must be implemented")

    def select(self, k):
        self.compute_selection_parameters()
        self.order_val_idx()
        return self.get_k_idx(k)


class RandomSelector(Selector):
    """
    Class to select random validation samples for next training
    """
    def order_val_idx(self):
        """
        Function that shuffle the validation indexes
        so that the first k indexes will be random
        """
        self.random_idx = self.val_idx
        np.random.shuffle(self.random_idx)

    def get_k_idx(self,k):
        return self.random_idx[:k].astype(int)


class LeastConfidenceSelector(Selector):
    """
    Class to select validation samples for next training
    with the least confidence sampling selection method
    """
    def compute_selection_parameters(self):
        """
        Function that computes the maximum confidence of each sample
        """
        # Sort the class probabilities of each sample
        sorted_probas = np.sort(self.outputs)

        self.maxprobas = sorted_probas[:,-1]

    def order_val_idx(self):
        """
        Function that order validation sample indexes
        based on the parameter of each sample
        """
        # Create an array containing index
        # and maximum class probability of each sample
        maxproba_with_idx = np.array([self.val_idx, self.maxprobas])
        
        # Get the smaller maximums class probabilities indexes
        # and then the associated sample indexes
        sort_idx = maxproba_with_idx[1,:].argsort()
        self.sample_idx =  maxproba_with_idx[0,sort_idx]

    def get_k_idx(self,k):
        return self.sample_idx[:k].astype(int)


class MarginSamplingSelector(Selector):
    """
    Class to select validation samples for next training
    with the margin sampling selection method
    """
    def compute_selection_parameters(self):
        """
        Function that computes the margin of each sample
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


class EntropySamplingSelector(Selector):
    """
    Class to select validation samples for next training
    with the uncertainty sampling selection method
    """

    def compute_selection_parameters(self):
        """
        Function that computes the uncertainty of each sample
        """
        # Sort the class probabilities of each sample
        sorted_probas = np.sort(self.outputs)
        ent = []
        for t in range(len(self.outputs)-1):
            print("1111111111111111111111111111111111")
            e = entropy(self.outputs[t])
            ent.append(e)
        self.parameters = ent

    def order_val_idx(self):
        """
        Function that order validation sample indexes
        based on the parameter of each sample
        """
        # Create an array containing index
        # and entropy of each sample
        ent_with_idx = np.array([self.val_idx, self.parameters])
        print(ent_with_idx.shape)
        # sort by entropy and get corresponding indexes
        seq = self.parameters
        sort_idx = sorted(range(len(seq)), key=seq.__getitem__)

        self.sample_idx = np.array(sort_idx)

    def get_k_idx(self, k):
        return self.sample_idx[:k].astype(int)


