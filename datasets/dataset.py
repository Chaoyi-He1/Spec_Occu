import os
import scipy.io as sio
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class Contrastive_data(Dataset):
    def __init__(self, data_path: str = "", cache: bool = True, 
                 past_steps: int = 32, future_steps: int = 12,
                 train_split: float = 0.8, train: bool = True) -> None:
        super(Contrastive_data, self).__init__()
        assert os.path.isfile(data_path), "path '{}' does not exist.".format(data_path)

        self.data_path = data_path
        self.train_split = train_split
        self.train = train
        self.data, self.data_len, self.train_len = self.cache_data(cache)

        self.future_step = future_steps
        self.past_steps = past_steps
        self.time_len = past_steps + future_steps
    
    def h5py_to_dict(self, h5_obj):
        if isinstance(h5_obj, h5py.File) or isinstance(h5_obj, h5py.Group):
            data = {}
            for key in h5_obj.keys():
                data[key] = self.h5py_to_dict(h5_obj[key])
            return data
        elif isinstance(h5_obj, h5py.Dataset):
            if len(h5_obj.shape) == 2:  # Check if the dataset represents a matrix
                return np.transpose(h5_obj[()])  # Transpose the matrix
            else:
                return h5_obj[()]
        else:
            return h5_obj

    def cache_data(self, cache: bool = True):
        print("Loading %s data from %s ..." % ("train" if self.train else "test", self.data_path))
        with h5py.File(self.data_path, 'r') as f:
            data = self.h5py_to_dict(f)
        if cache:
            data = np.stack([data["data_frame_I"], data["data_frame_Q"]], axis=1)
            train_len = int(data.shape[0] * self.train_split)
            data = data[:train_len, :, :] if self.train else data[train_len:, :, :]
            return data, data.shape[0], train_len
        else:
            train_len = int(data["data_frame_Q"].shape[0] * self.train_split)
            data_len = train_len if self.train else data["data_frame_I"].shape[0] - train_len
            return None, data_len, train_len

        
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
        index = self.train_len + index if not self.train else index
        if self.data is None:
            with h5py.File(self.data_path, "r") as f:
                real = f["data_frame_I"][index:index+self.time_len, :]
                imag = f["data_frame_Q"][index:index+self.time_len, :]
                data_past = np.stack([real[index:index+self.past_steps, :], 
                                      imag[index:index+self.past_steps, :]], axis=1)
                data_future = np.stack([real[index+self.past_steps:index+self.time_len, :],
                                        imag[index+self.past_steps:index+self.time_len, :]], axis=1)
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
