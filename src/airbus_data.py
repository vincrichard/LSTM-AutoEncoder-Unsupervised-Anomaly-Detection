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
    """Face Landmarks dataset."""

    def __init__(self, csv_file, seq_length, nrows=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file, delimiter=' ', nrows=nrows, header=None)
        self.seq_length = seq_length
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.create_seq(np.array(self.data.iloc[idx, :], dtype=np.float))
        return torch.from_numpy(data)

    def create_seq(self, row):
        inout_seq = row[:self.seq_length+1]
        for i in range(1, len(row) - self.seq_length -1):
            inout_seq = np.vstack((inout_seq, row[i:i+self.seq_length+1]))
        return inout_seq


class DataSeq(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]
