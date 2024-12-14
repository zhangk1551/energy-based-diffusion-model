import os
import pickle
import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from edm.utils import get_value

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TorchDriveEnvEpisodeDataset(Dataset):
    def __init__(self, data_dir, diffusion_keys, condition_keys, constraints=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.x_std = None
        self.x_mean = None
        self.data = []

        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "rb") as f:
                episode_data = pickle.load(f)
            if (constraints is not None) and ("location" in constraints) and (episode_data.location not in constraints["location"]):
                continue
#            for step_data in episode_data.step_data:
#                if len(diffusion_keys) == 1:
#                    x = get_value(diffusion_keys[0], step_data)
#                else:
#                    x = "_".join([get_value(key, step_data) for key in diffusion_keys])
#                if condition_keys is None:
#                    s = torch.empty(0)
#                elif len(condition_keys) == 1:
#                    s = get_value(condition_keys[0], step_data)
#                else:
#                    s = "_".join([get_value(key, step_data) for key in condition_keys])
#                self.data.append((x, s))

            for i in range(len(episode_data.step_data) - 3):
                step_data = episode_data.step_data[i: i+3]
                if len(diffusion_keys) == 1:
                    x = get_value(diffusion_keys[0], step_data)
                else:
                    x = "_".join([get_value(key, step_data) for key in diffusion_keys])
                if condition_keys is None:
                    s = torch.empty(0)
                elif len(condition_keys) == 1:
                    s = get_value(condition_keys[0], step_data)
                else:
                    s = "_".join([get_value(key, step_data) for key in condition_keys])
                self.data.append((x, s))

        self.x_dim = self.data[0][0].shape[-1]
        self.s_dim = self.data[0][1].shape if condition_keys is not None else None
        if (self.s_dim is not None) and (len(self.s_dim) == 1):
            self.s_dim = self.s_dim.item()
        if diffusion_keys == ['obs_birdview']:
            obs_birdviews = torch.stack([item[0] for item in self.data])
#            print("obs_birdviews shape")
#            print(obs_birdviews.shape)
            self.x_std, self.x_mean = torch.std_mean(obs_birdviews / 255.0, dim=(0, 2, 3))
            print("std: ", self.x_std)
            print("mean: ", self.x_mean)
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=self.x_mean, std=self.x_std)
            ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            x, s = self.data[idx]
            return self.transform(x), s
        return self.data[idx]


class EDMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.task = config.task
        self.diffusion_keys = config.diffusion_keys
        self.condition_keys = config.condition_keys
        self.constraints = config.data.constraints
        self.train_data_dir = config.data.train_data_dir
        self.val_data_dir = config.data.val_data_dir
        self.eval_obs_data_dirs = config.data.eval_obs_data_dirs
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.dataloader_num_workers

    def prepare_datasets(self):
        if self.task == "torchdriveenv":
            self.train_dataset = TorchDriveEnvEpisodeDataset(self.train_data_dir, self.diffusion_keys, self.condition_keys, self.constraints)
#            if self.val_data_dir is not None:
#                self.val_dataset = TorchDriveEnvEpisodeDataset(self.val_data_dir, self.diffusion_keys, self.condition_keys, self.constraints)
        self.size = len(self.train_dataset)
        self.x_dim = self.train_dataset.x_dim
        self.s_dim = self.train_dataset.s_dim
        self.x_mean = self.train_dataset.x_mean
        self.x_std = self.train_dataset.x_std
        self.x_transform = self.train_dataset.transform

        self.eval_data = self._read_eval_data()
        self.eval_obs_datasets = self._load_eval_obs_data()

    def _load_eval_obs_data(self):
        if self.eval_obs_data_dirs is None:
            return None
        datasets = []
        for data_dir in self.eval_obs_data_dirs:
            dataset = []
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if file_path[-4:] != ".pkl":
                    continue
                with open(file_path, "rb") as f:
                    obs_data = pickle.load(f)
                dataset.append(obs_data)
            datasets.append(dataset)
        return datasets

    def _read_eval_data(self):
        with open('data/test_eval_data/recurrent_state.pkl', 'rb') as f:
            recurrent_state = pickle.load(f)
        with open('data/test_eval_data/obs.pkl', 'rb') as f:
            obs = pickle.load(f)
        eval_data = {"recurrent_state": recurrent_state, "obs": obs}
        return eval_data

    def __len__(self):
        return self.size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          drop_last=True, num_workers=0)
