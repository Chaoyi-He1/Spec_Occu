import os
import scipy.io as sio
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class Contrastive_data(Dataset):
    def __init__(self, data_path: str = "", cache: bool = True, 
                 past_steps: int = 32, future_step: int = 12,
                 train_split: float = 0.8, train: bool = True) -> None:
        super(Contrastive_data, self).__init__()
        assert os.path.isfile(data_path), "path '{}' does not exist.".format(data_path)

        self.data_path = data_path
        self.train_split = train_split
        self.train = train
        self.data, self.data_len = self.cache_data(cache)

        self.future_step = future_step
        self.past_steps = past_steps
        self.time_len = past_steps + future_step

    def cache_data(self, cache: bool = True):
        data = sio.loadmat(self.data_path)
        if cache:
            data = np.stack([data["real"][:, np.newaxis, :], data["imag"][:, np.newaxis, :]], axis=1)
            train_len = (data.shape[0] * self.train_split) // 1
            data = data[:train_len, :, :] if self.train else data[train_len:, :, :]
            return data, data.shape[0]
        else:
            train_len = (data["real"].shape[0] * self.train_split) // 1
            data_len = train_len if self.train else data["real"].shape[0] - train_len
            return None, data_len

        
    def __len__(self):
        return self.data_len - self.time_len + 1
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data_past, data_future) where data_past is the past steps of the data and
            data_future is the future steps of the data to learn.
        """
        if self.data is None:
            with h5py.File(self.data_path, "r") as f:
                real = f["real"][index:index+self.time_len, :]
                imag = f["imag"][index:index+self.time_len, :]
                data_past = np.stack([real[index:index+self.past_steps, np.newaxis, :], 
                                      imag[index:index+self.past_steps, np.newaxis, :]], axis=1)
                data_future = np.stack([real[index+self.past_steps:index+self.time_len, np.newaxis, :],
                                        imag[index+self.past_steps:index+self.time_len, np.newaxis, :]], axis=1)
        else:
            data_past = self.data[index:index+self.past_steps, :, :]
            data_future = self.data[index+self.past_steps:index+self.time_len, :, :]
        
        return data_past, data_future
    
    @staticmethod
    def collate_fn(batch):
        data_past, data_future = list(zip(*batch))
        return torch.tensor(data_past), torch.tensor(data_future)

class Temporal_to_Freq_data(Dataset):
    def __init__(self, data_path: str = "", cache_data: bool = True, 
                 time_step: int = 12) -> None:
        super(Temporal_to_Freq_data, self).__init__()
        assert os.path.isfile(data_path), "path '{}' does not exist.".format(data_path)
        self.data_path = data_path
        self.data, self.label = self.cache_data() if cache_data else (None, None)
        self.time_step = time_step
    
    def cache_data(self):
        data = sio.loadmat(self.data_path)
        label = data["label"]
        data = np.stack([data["real"][:, np.newaxis, :], data["imag"][:, np.newaxis, :]], axis=1)
        assert data.shape[0] == label.shape[0], "data and label must have the same length."
        
        return data, label

    def __len__(self):
        if self.data is None:
            data = sio.loadmat(self.data_path)
            return data["real"].shape[0] - self.time_step + 1
        else:
            return self.data.shape[0] - self.time_step + 1
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, label) where data is the info within the time steps of the data and
            label is the frequency occupancy of the data within the time steps.
        """
        if self.data is None:
            with h5py.File(self.data_path, "r") as f:
                real = f["real"][index:index+self.time_step, :]
                imag = f["imag"][index:index+self.time_step, :]
                data = np.stack([real[:, np.newaxis, :], imag[:, np.newaxis, :]], axis=1)
                label = f["label"][index:index+self.time_step, :]
        else:
            data = self.data[index:index+self.time_step, :, :]
            label = self.label[index:index+self.time_step, :]
        
        assert data.shape[0] == label.shape[0], "data and label must have the same length."
        return data, label
    
    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        return torch.tensor(data), torch.tensor(label).long()
