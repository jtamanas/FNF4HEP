"""
Not necessary. TensorDataset does all the same shit
"""

from torch.utils.data import Dataset

class FairDataset(Dataset):
    def __init__(self, data, context, labels):
        self.data = data
        self.context = context
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.context[idx], self.labels[idx]
