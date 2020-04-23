import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class AirbusData(Dataset):
    def __init__(self, path, type, nrows=None):
        """
        Args:
            path (string): Path to file with annotations.
            type (string): 'csv' or 'pytorch'
        """
        if type == 'csv':
            self.data = pd.read_csv(path, delimiter=' ', nrows=nrows, header=None)
        if type == 'pytorch':
            self.data = torch.load(path)
        else:
            raise ValueError('type value is wrong: ', type)
        self.type = type

    def __len__(self):
        if self.type == 'csv':
            return len(self.data)
        if self.type == 'pytorch':
            return self.data.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.type == 'csv':
            return torch.from_numpy(np.array(self.data.iloc[idx, :], dtype=np.float))
        if self.type == 'pytorch':
            return self.data[:,idx,:]


# ---  Don't use what is down there  ---


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
