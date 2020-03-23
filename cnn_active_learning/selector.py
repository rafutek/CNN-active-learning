import numpy as np


class Selector(object):
    def select(self):
        pass
    def indexes(self):
        pass

class RandomSelector(Selector):
    @staticmethod
    def select(network_output, num_loop, k, selection):
        batch_size = len(network_output)
        image_idx = np.arange(batch_size) + k + batch_size*num_loop # samples index in the pool
        if selection is None:
            selection = image_idx
        else:
            selection = np.concatenate((selection,image_idx))
            selection = selection[:k]
        return selection

    @staticmethod
    def indexes(selection):
        return selection.astype(int)

class UncertaintySelector(Selector):
    @staticmethod
    def select(network_output, num_loop, k, selection):
        batch_size = len(network_output)
        image_idx = np.arange(batch_size) + k + batch_size*num_loop # samples index in the pool
        maxprobability, _ = network_output.max(dim=1)
        batch_maxprobas = np.array([maxprobability.tolist(), image_idx.tolist()])
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

