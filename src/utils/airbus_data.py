import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class AirbusData(Dataset):
    def __init__(self, csv_file, nrows=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file, delimiter=' ', nrows=nrows, header=None)
        self.transform = transform
        if transform is not None:
            self.data = self.data.apply(self.transform, axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(np.array(self.data.iloc[idx, :], dtype=np.float))


class AirbusDataSeq(Dataset):

    def __init__(self, path, list_id):
        self.path = path
        self.list_id = list_id

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.load(self.path + str(idx) + '.pt')


class DataSeq(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

def create_seq(row, seq_length):
    inout_seq = row[:seq_length+1]
    for i in range(1, len(row) - seq_length -1):
        inout_seq = np.vstack((inout_seq, row[i:i+seq_length+1]))
    return inout_seq
