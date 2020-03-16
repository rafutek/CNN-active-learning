from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class Dataset(Dataset):
    def __init__(self, samples, labels):
        if len(samples) ==  len(labels):
            self.samples = samples
            self.labels = labels
        else:
            raise AttributeError('samples and labels must be of the same length')
        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, label = self.samples[index], self.labels[index]
        sample = torch.tensor(sample, dtype=torch.float)
        sample = self.transform(sample)
        return sample, label



