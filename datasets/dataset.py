import os
import scipy.io as sio
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm


class Contrastive_data(Dataset):
    def __init__(self, data_path: str = "", cache: bool = True, past_steps: int = 32,
                 future_steps: int = 12, train_split: float = 0.8, train: bool = True,
                 num_frames_per_clip: int = 256, temp_dim: int = 1024) -> None:
        super(Contrastive_data, self).__init__()
        assert os.path.isfile(data_path), "path '{}' does not exist.".format(data_path)

        self.data_path = data_path
        self.train_split = train_split
        self.train = train

        self.future_step = future_steps
        self.past_steps = past_steps
        self.time_len = past_steps + future_steps

        self.temp_dim = temp_dim
        self.num_frames_per_clip = num_frames_per_clip

        self.data, self.data_len, self.train_len = self.cache_data(cache)

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
        # for k, v in data.items():
        #     data[k] = v[:100000, :]
        if cache:
            data = np.stack([data["data_frame_I"], data["data_frame_Q"]], axis=1)
            train_len = int(data.shape[0] * self.train_split)
            data = data[:train_len, :, :] if self.train else data[train_len:, :, :]
            data = data[:-(data.shape[0] % self.num_frames_per_clip), :, :]

            data = np.nan_to_num(data, nan=0.)
            data = np.reshape(data, (-1, self.num_frames_per_clip, 2, self.temp_dim))
            data = data.transpose(0, 2, 1, 3).copy()
            return data, \
                   data.shape[0], int(train_len // self.num_frames_per_clip)
        else:
            train_len = int(data["data_frame_Q"].shape[0] * self.train_split)
            data_len = train_len if self.train else data["data_frame_I"].shape[0] - train_len
            return None, int(data_len // self.num_frames_per_clip), int(train_len // self.num_frames_per_clip)

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
            index = (self.train_len + index) * self.num_frames_per_clip \
                if not self.train else index * self.num_frames_per_clip
            with h5py.File(self.data_path, "r") as f:
                real = f["data_frame_I"][index:index +
                                               self.time_len * self.num_frames_per_clip, :]
                real = np.reshape(real, (self.time_len, self.num_frames_per_clip, self.temp_dim))
                imag = f["data_frame_Q"][index:index +
                                               self.time_len * self.num_frames_per_clip, :]
                imag = np.reshape(imag, (self.time_len, self.num_frames_per_clip, self.temp_dim))
                data_past = np.stack([real[index:index + self.past_steps, :, :],
                                      imag[index:index + self.past_steps, :, :]], axis=1)
                data_future = np.stack([real[index + self.past_steps:index + self.time_len, :, :],
                                        imag[index + self.past_steps:index + self.time_len, :, :]], axis=1)

            return np.nan_to_num(data_past, nan=0.), np.nan_to_num(data_future, nan=0.)

        else:
            data_past = self.data[index:index + self.past_steps, :, :, :]
            data_future = self.data[index + self.past_steps:index + self.time_len, :, :, :]

        return data_past, data_future

    @staticmethod
    def collate_fn(batch):
        data_past, data_future = list(zip(*batch))
        data_past = torch.tensor(np.array(data_past)).float()
        data_future = torch.tensor(np.array(data_future)).float()

        data_past_c = data_past[:, :, 0, :, :] + 1j * data_past[:, :, 1, :, :]
        data_future_c = data_future[:, :, 0, :, :] + 1j * data_future[:, :, 1, :, :]

        data_past_f = torch.fft.fft(data_past_c, dim=-1)
        data_future_f = torch.fft.fft(data_future_c, dim=-1)

        data_past_f = torch.stack([data_past_f.real, data_past_f.imag], dim=2)
        data_future_f = torch.stack([data_future_f.real, data_future_f.imag], dim=2)

        data_past = torch.cat([data_past, data_past_f], dim=2)
        data_future = torch.cat([data_future_f, data_future_f], dim=2)

        return data_past, data_future


class Contrastive_data_multi_env(Dataset):
    """
    Dataset for contrastive learning with multiple wireless environments.
    The data is stored in a folder, each file is a wireless environment.
    Each file contains the rx signal for I and Q channel.
    One sampled timeframe is stored in one row
    """

    def __init__(self, data_folder_path: str = "", cache: bool = True, in_type: str = "1d",
                 past_steps: int = 32, future_steps: int = 12, train: bool = True,
                 num_frames_per_clip: int = 256, temp_dim: int = 1024) -> None:
        super(Contrastive_data_multi_env, self).__init__()

        assert os.path.isdir(data_folder_path), "path '{}' does not exist.".format(data_folder_path)
        self.data_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path)]
        self.data_files.sort()
        self.num_env = len(self.data_files)
        self.in_type = in_type

        self.cache = cache
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.train = train
        self.num_frames_per_clip = num_frames_per_clip
        self.temp_dim = temp_dim

        self.data_dict, self.data_len, self.min_len = self.cache_data()

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

    def preprocess_data(self, data):
        """
        Preprocess the data to reshape it as a 3D input for the model.
        The raw data was stored as a 2D matrix, each row represents a sampled timeframe.
        Raw data shape: (num_frames, 2, temprol_dim)
        Preprocessed data shape: (-1, 4, num_frames_per_clip, temp_dim) if in_type is "2d"
        Preprocessed data shape: (-1, 4, temp_dim) if in_type is "1d"

        Args:
            h5py_data: data from h5py file, a dict contains I and Q channel
        Returns:
            data: reshaped data concatenated with its FFT
        """
        if isinstance(data, dict):
            # if the data is from h5py file, it means that the data is cached,
            # the dict contains I and Q channel, each channel is a 2D matrix
            # with shape (num_frames, temporal_dim)
            real = data["data_frame_I"][:51200, :].astype(np.float32)
            imag = data["data_frame_Q"][:51200, :].astype(np.float32)
            data = np.stack([real, imag], axis=1)
            # data_c = real + 1j * imag
            # data_fft = np.fft.fft(data_c, axis=-1)
            # data_f_real = np.real(data_fft)
            # data_f_imag = np.imag(data_fft)
            # data = np.concatenate([data, 
            #                     data_f_real[:, np.newaxis, :], 
            #                     data_f_imag[:, np.newaxis, :]], axis=1)
            # data = data[:-(data.shape[0] % self.num_frames_per_clip), :, :]
            if self.in_type == "2d":
                data = data.reshape(-1, self.num_frames_per_clip, 2, self.temp_dim). \
                    transpose(0, 2, 1, 3)
            # data = data.astype(np.float32)
        elif isinstance(data, np.ndarray):
            # if the data is a numpy array, it means that the data is not cached,
            # the data is a 4D matrix with shape (batch, num_frames, 2, temporal_dim)
            # the data is read by __getitem__ function
            # and the preprocessing is called by collate_fn function

            real = data[:, :, 0, :]
            imag = data[:, :, 1, :]
            data_c = real + 1j * imag
            data_fft = np.fft.fft(data_c, axis=-1)
            data_f_real = np.real(data_fft)
            data_f_imag = np.imag(data_fft)
            data = np.concatenate([data,
                                   data_f_real[:, :, np.newaxis, :],
                                   data_f_imag[:, :, np.newaxis, :]], axis=2)
            # data = data[:, :-(data.shape[0] % self.num_frames_per_clip), :, :]
            if self.in_type == "2d":
                b, _, _, _ = data.shape
                data = data.reshape(b, -1, self.num_frames_per_clip, 4, self.temp_dim). \
                    transpose(0, 1, 3, 2, 4)
            data = data.astype(np.float32)

        return data.copy()

    def cache_data(self):
        data_dict = {}
        data_len = {}
        min_len = np.inf
        for i, data_file_path in tqdm(enumerate(self.data_files)):
            print("Loading data from %s ... (%d / %d)" %
                  (data_file_path, i, len(self.data_files)))
            if i not in [0, 1, 2, 3, 4, 5]:
                continue
            with h5py.File(data_file_path, 'r') as f:
                data = self.h5py_to_dict(f)
            if self.cache:
                data = self.preprocess_data(data)
                data_dict[i] = data
                data_len[i] = data.shape[0] - self.past_steps - self.future_steps + 1
            else:
                data_dict[i] = None
                data_len[i] = data["data_frame_I"].shape[0] // self.num_frames_per_clip \
                              - self.past_steps - self.future_steps + 1 if self.in_type == "2d" \
                    else data["data_frame_I"].shape[0] - self.past_steps - self.future_steps + 1
            min_len = min(min_len, data_len[i])
        return data_dict, data_len, min_len

    def __len__(self):
        return len(self.data_len)

    def __getitem__(self, index):
        """
        The data is stored in a folder, index is the index of the file.
        If the data is cached, the data is a 4D matrix with shape (num_frames, 4, num_frames_per_clip, temp_dim)
        If the data is not cached, the data is a dict contains I and Q channel, each channel is a 2D matrix
        Args:
            index (int): Index
        Returns:
            tuple: (data_past, data_future) where data_past is the past data and
            data_future is the future data.
        """
        index = index % len(self.data_files)
        time_step = np.random.randint(self.data_len[index])
        if self.cache:
            data = self.data_dict[index]
            data_past = data[time_step:time_step + self.past_steps, ...]
            data_future = data[time_step + self.past_steps:
                               time_step + self.past_steps + self.future_steps, ...]
        else:
            with h5py.File(self.data_files[index], 'r') as f:
                data = self.h5py_to_dict(f)
            data = np.stack([data["data_frame_I"], data["data_frame_Q"]], axis=1)
            if self.in_type == "2d":
                data = data[time_step * self.num_frames_per_clip:
                            (time_step + self.past_steps + self.future_steps) * self.num_frames_per_clip, :, :]
            else:
                data = data[time_step:time_step + self.past_steps + self.future_steps, :, :]
            data = self.preprocess_data(data)
            data_past = data[:self.past_steps * self.num_frames_per_clip, ...]
            data_future = data[self.past_steps * self.num_frames_per_clip:, ...]
        return data_past, data_future, index

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: list of tuples (data_past, data_future).
        Returns:
            tuple: (data_past, data_future) where data_past is the past data and
            data_future is the future data.
        """
        data_past, data_future, index = list(zip(*batch))
        data_past = np.stack(data_past, axis=0)
        data_future = np.stack(data_future, axis=0)
        index = np.array(index)
        data_past = torch.from_numpy(data_past).float()
        data_future = torch.from_numpy(data_future).float()
        index = torch.from_numpy(index).int()
        return data_past, data_future, index


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
                real = f["real"][index:index + self.time_step, :]
                imag = f["imag"][index:index + self.time_step, :]
                data = np.stack([real[:, np.newaxis, :], imag[:, np.newaxis, :]], axis=1)
                label = f["label"][index:index + self.time_step, :]
        else:
            data = self.data[index:index + self.time_step, :, :]
            label = self.label[index:index + self.time_step, :]

        assert data.shape[0] == label.shape[0], "data and label must have the same length."
        return data, label

    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        return torch.tensor(data), torch.tensor(label).long()


class Temporal_to_Freq_data_multi_env(Dataset):
    def __init__(self, data_folder_path: str = "", cache: bool = True,
                 time_step: int = 12) -> None:
        super(Temporal_to_Freq_data_multi_env, self).__init__()
        assert os.path.isdir(data_folder_path), "path '{}' does not exist.".format(data_folder_path)
        self.data_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path)]
        self.data_files.sort()
        self.num_env = len(self.data_files)
        self.cache = cache
        self.time_step = time_step
        (self.data_dict, self.label_dict,
         self.data_len, self.min_len) = self.cache_data()

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

    def cache_data(self):
        data_dict = {}
        label_dict = {}
        data_len = {}
        min_len = np.inf
        for i, data_file_path in enumerate(self.data_files):
            print("Loading data from %s ... (%d / %d)" %
                  (data_file_path, i, len(self.data_files)))
            # if ("20" not in data_file_path) and ("21" not in data_file_path):
            #     continue
            # continue
            with h5py.File(data_file_path, 'r') as f:
                data = self.h5py_to_dict(f)

            if self.cache:
                start_index = np.random.randint(0,
                                                data["data_frame_I"].shape[0] - 20000)
                label = data["label_frame"][start_index:start_index + 10000, :]
                data = np.stack([data["data_frame_I"][start_index:start_index + 10000, :],
                                 data["data_frame_Q"][start_index:start_index + 10000, :]], axis=1)
                assert data.shape[0] == label.shape[0], "data and label must have the same length."
                data_dict[i] = data.astype(np.float16)
                label_dict[i] = label.astype(int)
                data_len[i] = data_dict[i].shape[0] - self.time_step + 1
            else:
                data_dict[i] = None
                data_len[i] = data["data_frame_I"].shape[0] - self.time_step + 1
                label_dict[i] = data["label_frame"]
            min_len = min(min_len, data_len[i])
        return data_dict, label_dict, data_len, min_len

    def __len__(self):
        return len(self.label_dict) * 768

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, label) where data is the info within the time steps of the data and
            label is the frequency occupancy of the data within the time steps.
        """
        # index = 12
        index = index % len(self.label_dict)
        time_step = np.random.randint(self.data_len[index])
        if self.data_dict is None:
            with h5py.File(self.data_files[index], 'r') as f:
                data = self.h5py_to_dict(f)
            real = data["data_frame_I"][time_step:time_step + self.time_step, :]
            imag = data["data_frame_Q"][time_step:time_step + self.time_step, :]
            label = data["label_frame"][time_step:time_step + self.time_step, :]
            data = np.stack([real, imag], axis=1)
        else:
            data = self.data_dict[index][time_step:time_step + self.time_step, ...]
            label = self.label_dict[index][time_step:time_step + self.time_step, ...]
        return data, label

    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        data = np.stack(data, axis=0)
        label = np.stack(label, axis=0)
        return torch.from_numpy(data).float(), \
               torch.from_numpy(label).float()


class Diffusion_multi_env(Dataset):
    def __init__(self, data_folder_path: str = "", cache: bool = True,
                 past_steps: int = 32, future_steps: int = 12, train: bool = True,
                 temp_dim: int = 1024) -> None:
        super(Diffusion_multi_env, self).__init__()
        assert os.path.isdir(data_folder_path), "path '{}' does not exist.".format(data_folder_path)
        self.data_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path)]
        self.data_files.sort()
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.train = train
        self.temp_dim = temp_dim
        self.cache = cache
        self.total_time_steps = past_steps + future_steps

        (self.data_dict, self.label_dict,
         self.data_len, self.min_len) = self.cache_data()

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

    def cache_data(self):
        data_dict = {}
        label_dict = {}
        data_len = {}
        min_len = np.inf
        for i, data_file_path in enumerate(self.data_files):
            print("Loading data from %s ... (%d / %d)" %
                  (data_file_path, i, len(self.data_files)))
            # if i != 0:
            #     continue

            with h5py.File(data_file_path, 'r') as f:
                data = self.h5py_to_dict(f)

            if self.cache:
                start_index = np.random.randint(0,
                                                data["data_frame_I"].shape[0] - 20000)
                label = data["label_frame"][start_index:start_index + 10000, ...]
                data = np.stack([data["data_frame_I"][start_index:start_index + 10000, ...],
                                 data["data_frame_Q"][start_index:start_index + 10000, ...]], axis=1)
                assert data.shape[0] == label.shape[0], "data and label must have the same length."
                data_dict[i] = (data * 50).astype(np.float16)
                label_dict[i] = label.astype(int)
                data_len[i] = data_dict[i].shape[0] - self.total_time_steps + 1
            else:
                data_dict[i] = None
                data_len[i] = data["data_frame_I"].shape[0] - self.total_time_steps + 1
                label_dict[i] = data["label_frame"]
            min_len = min(min_len, data_len[i])
        return data_dict, label_dict, data_len, min_len

    def __len__(self):
        return len(self.label_dict) * 192 if self.train \
            else len(self.label_dict) * 60

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, label) where data is the info within the time steps of the data and
            label is the frequency occupancy of the data within the time steps.
        """
        index = index % len(self.label_dict)
        # index = 0
        time_step = np.random.randint(self.data_len[index])
        if self.data_dict is None:
            with h5py.File(self.data_files[index], 'r') as f:
                data = self.h5py_to_dict(f)
            real = data["data_frame_I"][time_step:time_step + self.total_time_steps, :]
            imag = data["data_frame_Q"][time_step:time_step + self.total_time_steps, :]
            label = data["label_frame"][time_step:time_step + self.total_time_steps, :]
            data = np.stack([real, imag], axis=1)
        else:
            data = self.data_dict[index][time_step:time_step + self.total_time_steps, ...]
            label = self.label_dict[index][time_step:time_step + self.total_time_steps, ...]

        past_data = data[:self.past_steps, ...]
        future_data = data[self.past_steps:, ...]
        past_label = label[:self.past_steps, ...]
        future_label = label[self.past_steps:, ...]
        return past_data, past_label, future_data, future_label

    @staticmethod
    def collate_fn(batch):
        past_data, past_label, future_data, future_label = list(zip(*batch))
        past_data = np.stack(past_data, axis=0)
        past_label = np.stack(past_label, axis=0)
        future_data = np.stack(future_data, axis=0)
        future_label = np.stack(future_label, axis=0)
        return torch.from_numpy(past_data).float(), \
               torch.from_numpy(past_label).float(), \
               torch.from_numpy(future_data).float(), \
               torch.from_numpy(future_label).float()


class CPC_multi_env_test(Dataset):
    def __init__(self, data_folder_path: str = "", cache: bool = True,
                 past_steps: int = 32, future_steps: int = 12, train: bool = True,
                 temp_dim: int = 1024) -> None:
        super(CPC_multi_env_test, self).__init__()
        assert os.path.isdir(data_folder_path), "path '{}' does not exist.".format(data_folder_path)
        self.data_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path)]
        self.data_files.sort()
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.train = train
        self.temp_dim = temp_dim
        self.cache = cache
        self.total_time_steps = past_steps + future_steps

        (self.data_dict, self.label_dict,
         self.data_len, self.min_len) = self.cache_data()

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

    def cache_data(self):
        data_dict = {}
        label_dict = {}
        data_len = {}
        min_len = np.inf
        for i, data_file_path in enumerate(self.data_files):
            print("Loading data from %s ... (%d / %d)" %
                  (data_file_path, i, len(self.data_files)))
            # if i != 0:
            #     continue

            with h5py.File(data_file_path, 'r') as f:
                data = self.h5py_to_dict(f)

            if self.cache:
                start_index = np.random.randint(0,
                                                data["data_frame_I"].shape[0] - 20000)
                label = data["label_frame"][start_index:start_index + 10000, ...]
                data = np.stack([data["data_frame_I"][start_index:start_index + 10000, ...],
                                 data["data_frame_Q"][start_index:start_index + 10000, ...]], axis=1)
                assert data.shape[0] == label.shape[0], "data and label must have the same length."
                data_dict[i] = (data * 50).astype(np.float16)
                label_dict[i] = label.astype(int)
                data_len[i] = data_dict[i].shape[0] - self.total_time_steps + 1
            else:
                data_dict[i] = None
                data_len[i] = data["data_frame_I"].shape[0] - self.total_time_steps + 1
                label_dict[i] = data["label_frame"]
            min_len = min(min_len, data_len[i])
        return data_dict, label_dict, data_len, min_len

    def __len__(self):
        return len(self.label_dict) * 192 if self.train \
            else len(self.label_dict) * 60

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, label) where data is the info within the time steps of the data and
            label is the frequency occupancy of the data within the time steps.
        """
        index = index % len(self.label_dict)
        # index = 0
        time_step = np.random.randint(self.data_len[index])
        if self.data_dict is None:
            with h5py.File(self.data_files[index], 'r') as f:
                data = self.h5py_to_dict(f)
            real = data["data_frame_I"][time_step:time_step + self.total_time_steps, :]
            imag = data["data_frame_Q"][time_step:time_step + self.total_time_steps, :]
            label = data["label_frame"][time_step:time_step + self.total_time_steps, :]
            data = np.stack([real, imag], axis=1)
        else:
            data = self.data_dict[index][time_step:time_step + self.total_time_steps, ...]
            label = self.label_dict[index][time_step:time_step + self.total_time_steps, ...]

        past_data = data[:self.past_steps, ...]
        future_data = data[self.past_steps:, ...]
        past_label = label[:self.past_steps, ...]
        future_label = label[self.past_steps:, ...]
        return past_data, past_label, future_data, future_label, index

    @staticmethod
    def collate_fn(batch):
        past_data, past_label, future_data, future_label, index = list(zip(*batch))
        past_data = np.stack(past_data, axis=0)
        past_label = np.stack(past_label, axis=0)
        future_data = np.stack(future_data, axis=0)
        future_label = np.stack(future_label, axis=0)
        index = np.array(index)
        return torch.from_numpy(past_data).float(), \
               torch.from_numpy(past_label).float(), \
               torch.from_numpy(future_data).float(), \
               torch.from_numpy(future_label).float(), \
               torch.from_numpy(index).int()