from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class Dataset(Dataset):
    """
    Class to create a torch Dataset from local data
    """
    def __init__(self, samples, labels):
        """
        Constructor that set dataset variables
        Args:
            samples: array containing the dataset samples
            labels: array containing the label of each sample
        """
        if len(samples) ==  len(labels):
            self.samples = samples
            self.labels = labels
        else:
            raise AttributeError('samples and labels must be of the same length')
        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        """
        Function to get the length of the dataset
        Returns:
            The length of the dataset
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Function to get an item of the dataset
        Args:
            index: the position of the dataset item
        Returns:
            sample: the associated sample as a normalized tensor
            label: the associated label
        """
        sample, label = self.samples[index], self.labels[index]
        sample = torch.tensor(sample, dtype=torch.float)
        sample = self.transform(sample)
        return sample, label



