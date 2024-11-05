import os
import pickle
import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TorchDriveEnvEpisodeDataset(Dataset):
    def __init__(self, data_dir, diffusion_keys, condition_keys, constraints=None):
        super().__init__()
        self.data_dir = data_dir

        self.data = []

        def get_value(key, step_data):
            if key == "obs_birdview":
                return step_data.obs_birdview.squeeze()
            if key == "recurrent_state":
                return torch.Tensor(step_data.recurrent_states[0]).squeeze() # .cuda()
            if key == "action":
                return step_data.ego_action.squeeze()

        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "rb") as f:
                episode_data = pickle.load(f)
            if (constraints is not None) and ("location" in constraints) and (episode_data.location not in constraints["location"]):
                continue
            for step_data in episode_data.step_data:
                if len(diffusion_keys) == 1:
                    x = get_value(diffusion_keys[0], step_data)
                else:
                    x = "_".join([get_value(key, step_data) for key in diffusion_keys])
                if condition_keys is None:
                    s = None
                elif len(condition_keys) == 1:
                    s = get_value(condition_keys[0], step_data)
                else:
                    s = "_".join([get_value(key, step_data) for key in condition_keys])
                self.data.append((x, s))

        self.x_dim = self.data[0][0].shape[-1]
        self.s_dim = self.data[0][1].shape[-1] if condition_keys is not None else None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.dataloader_num_workers

    def prepare_datasets(self):
        if self.task == "torchdriveenv":
            self.train_dataset = TorchDriveEnvEpisodeDataset(self.train_data_dir, self.diffusion_keys, self.condition_keys, self.constraints)
            if self.val_data_dir is not None:
                self.val_dataset = TorchDriveEnvEpisodeDataset(self.val_data_dir, self.diffusion_keys, self.condition_keys, self.constraints)
        self.size = len(self.train_dataset)
        self.x_dim = self.train_dataset.x_dim
        self.s_dim = self.train_dataset.s_dim

        self.eval_data = self._read_eval_data()

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

#    def val_dataloader(self):
#        return DataLoader(self.val_dataset, batch_size=self.batch_size,
#                          drop_last=True, num_workers=0)

# class DataSampler():
#     def __init__(self, data, device):
#         self.data = {}
#         for key in data:
#             self.data[key] = torch.from_numpy(data[key]).float()
# 
#         self.size = next(iter(data.values())).shape[0]
#         self.device = device
# 
#     def sample(self, diffusion_keys, batch_size, concat=False):
#         ind = torch.randint(0, self.size, size=(batch_size,))
#         samples = {}
#         for key in diffusion_keys:
#           samples[key] = self.data[key][ind].to(self.device)
# 
#         if concat:
#             return torch.cat(tuple(samples.values()), dim=-1)
#         return samples
# 
