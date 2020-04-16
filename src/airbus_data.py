import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class AirbusData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, nrows=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file, delimiter=' ', nrows=nrows)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(np.array(self.data.iloc[idx,:], dtype=np.float))
