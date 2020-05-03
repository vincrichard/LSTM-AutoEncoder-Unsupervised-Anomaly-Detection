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
        elif type == 'pytorch':
            self.data = torch.load(path)
        else:
            raise ValueError('type value is wrong: ', type)
        self.type = type

    def __len__(self):
        if self.type == 'csv':
            return len(self.data)
        if self.type == 'pytorch':
            return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.type == 'csv':
            return torch.from_numpy(np.array(self.data.iloc[idx, :], dtype=np.float))
        if self.type == 'pytorch':
            return self.data[idx,:,:]