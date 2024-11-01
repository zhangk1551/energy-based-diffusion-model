import os
import pickle
import logging
import pandas as pd

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from collisions_blame.utils import read_video, get_location, extract_states, \
    mturk_agent_binary, mturk_agent_binary_score, \
    mturk_agent_multiclass, mturk_agent_multiclass_score, mturk_reasons, \
    read_label
from collisions_blame.configs import ModuleType, Task


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CollisionsCollection:
    def __init__(self, config):
        self.data_dir = config.data.data_dir
        self.collisions = {}

        if config.data.mturk_file is not None:
            mturk_df = pd.read_csv(config.data.mturk_file)
            mturk_data = mturk_df.set_index("video").to_dict("index")
        else:
            mturk_data = None

        if config.data.label_file is not None:
            with open(config.data.label_file, 'rb') as f:
                label_data = pickle.load(f)
        else:
            label_data = None

        for collision_file in os.listdir(self.data_dir):
            if collision_file[-4:] == ".pkl":
                try:
                    key = collision_file[:-4]
                    if (mturk_data is not None) and (key not in mturk_data):
                        continue

                    if (label_data is not None) and (key not in label_data):
                        continue

                    with open(os.path.join(self.data_dir, collision_file), 'rb') as f:
                        collision = pickle.load(f)

                    if (mturk_data is not None) and key in mturk_data:
                        collision["mturk"] = mturk_data[key]

                    if (label_data is not None) and key in label_data:
                        collision["label"] = label_data[key]

                    collision = self._flip_collision(collision)
                    collision = self._move_to_cpu(collision)

                    if config.data.ego_birdview_res is not None:
                        collision["agent_states"] = self.calc_agent_states(key, collision, config.data.resolution)
                        collision["agent_psi"] = self.calc_agent_psi(collision)

                    self.collisions[key] = collision
                    if len(self.collisions) % 10 == 0:
                        logger.info(f"Processed {len(self.collisions)} collisions")

                except Exception as e:
                    logger.warning(f"error pickle file: {collision_file}: \n{e}")

        logger.info(
            f"CollisionsCollection in {self.data_dir}: {len(self.collisions)} collisions")

    def _flip_collision(self, collision):
        def _switch_pair(x):
            return (x[1], x[0])
        if collision["highlight_colors"][0][0] != 255:
            # other keys: collsion_timesteps, collision_centers, world_center, states
            keys = ["collision_agents", "collision_positions", "collision_agents",
                    "offroad_infraction", "wrong_way_infraction", "traffic_lights_infraction",
                    "recurrent_states"]
            for key in keys:
                collision[key] = _switch_pair(collision[key])
        collision.pop("highlight_colors")
        return collision

    def _move_to_cpu(self, collision):
        collision["collision_agents"] = (collision["collision_agents"][0].cpu(),
                                         collision["collision_agents"][1].cpu())
        collision["recurrent_states"] = (collision["recurrent_states"][0].cpu(),
                                         collision["recurrent_states"][1].cpu())
        return collision

    def calc_agent_states(self, key, collision, resolution):
        try:
            with open(os.path.join(self.data_dir, key + "_states.pickle"), "rb") as states_f:
                agent_states = pickle.load(states_f)
        except:
            frames = read_video(
                os.path.join(self.data_dir, key + ".mp4"), end_frame=collision["collsion_timesteps"],
                resolution=resolution)
            agent_states = extract_states(frames)
            with open(os.path.join(self.data_dir, key + "_states.pickle"), 'wb') as states_f:
                pickle.dump(agent_states, states_f)
        return agent_states

    def calc_agent_psi(self, collision):
        purple_agent = collision["collision_agents"][0].detach()
        yellow_agent = collision["collision_agents"][1].detach()
        return {"purple": collision["states"][purple_agent, :, 2].cpu(),
                "yellow": collision["states"][yellow_agent, :, 2].cpu()}

    def __len__(self):
        return len(self.collisions)


class ConditionedCollisionsDataset(Dataset):
    def __init__(self, config, condition, collisions):
        super().__init__()
        self.data_dir = config.data.data_dir
        self.resolution = config.data.resolution
        self.raw = config.data.raw
        self.use_states = config.data.use_states
        self.ego_birdview_res = config.data.ego_birdview_res
        self.keys = []
        self.key_to_idx = {}
        self.collisions = collisions

        logger.info(
            f"Processing ConditionedCollisionsDataset: {condition.__name__}")

        self.labels = []
        for key, collision in collisions.items():
            label = condition(collision)
            if label is None:
                continue
            self.labels.append(label)
            self.key_to_idx[key] = len(self.keys)
            self.keys.append(key)

        logger.info(f"Dataset length: {len(self.keys)}")

    def produce_input(self, idx):
        key = self.keys[idx]
        collision = self.collisions[key]
        if self.raw:
            return read_video(os.path.join(self.data_dir, key + ".mp4"),
                              end_frame=collision["collsion_timesteps"],
                              agent_states=collision["agent_states"] if self.ego_birdview_res is not None else None,
                              agent_psi=collision["agent_psi"] if self.ego_birdview_res is not None else None,
                              ego_birdview_res=self.ego_birdview_res,
                              resolution=self.resolution)
        if self.use_states:
            pass
        return torch.stack(collision["recurrent_states"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.produce_input(idx), self.labels[idx]

    def get_video(self, idx, display=False):
        key = self.keys[idx]
        collision = self.collisions[key]
        return read_video(os.path.join(self.data_dir, key + ".mp4"),
                          end_frame=collision["collsion_timesteps"], display=display)

    def get_video_url(self, idx):
        return os.path.join(self.data_dir, self.keys[idx] + ".mp4")


def split_dataset_according_to_locations(keys, dataset, with_keys=False):
    val_locations = ["cambie_broadway_2", "davie_street_and_marinaside_crescent", "trans_canada_and_king_edward",
                     "revelstoke_victoria_and_wright", "miguel_aleman_and_palmas_acapulco", "trans_canada_and_lougheed",
                     "cuautehmoc_avenue_acapulco_3", "16th_at_wesbrook_mall"]
    test_locations = ["revelstoke_victoria_and_wright_2", "trans_canada_and_coleman_4", "cambie_broadway",
                      "eburne_park_3", "west_16th_1", "trans_canada_and_capilano",
                      "west_16th", "dunbar_diversion"]
    train_indice = []
    val_indice = []
    test_indice = []
    if with_keys:
        train_keys = []
        val_keys = []
        test_keys = []
    for idx, key in enumerate(keys):
        location = get_location(key)
        if location in val_locations:
            val_indice.append(idx)
            if with_keys:
                val_keys.append(key)
        elif location in test_locations:
            test_indice.append(idx)
            if with_keys:
                test_keys.append(key)
        else:
            train_indice.append(idx)
            if with_keys:
                train_keys.append(key)

    train_dataset = Subset(dataset, train_indice)
    val_dataset = Subset(dataset, val_indice)
    test_dataset = Subset(dataset, test_indice)

    if with_keys:
        return train_dataset, val_dataset, test_dataset, train_keys, val_keys, test_keys
    else:
        return train_dataset, val_dataset, test_dataset


class CollisionsDataModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.dataloader_num_workers
        self.is_random_split = config.data.is_random_split
        self.num_classes = config.module.num_classes
        self.weighted_sampling = config.module.weighted_sampling

        self.condition = self._decide_condition()

    def _decide_condition(self):
        if self.config.data.label_file is not None:
            return read_label

        module_config = self.config.module
        if module_config.predict_reason:
            return mturk_reasons
        if module_config.task == Task.binary:
            if module_config.module_type == ModuleType.classification:
                return mturk_agent_binary
            if module_config.module_type == ModuleType.regression:
                return mturk_agent_binary_score
        if module_config.task == Task.multiclass:
            if module_config.module_type == ModuleType.classification:
                return mturk_agent_multiclass
            if module_config.module_type == ModuleType.regression:
                return mturk_agent_multiclass_score

    def prepare_datasets(self):
        self.collisions = CollisionsCollection(self.config).collisions

        self.mturk_dataset = ConditionedCollisionsDataset(self.config,
                                                          self.condition,
                                                          self.collisions)

        self.dataset = self.mturk_dataset
        self.size = len(self.dataset)

        # split train / val / test
        if self.is_random_split:
            train_ratio = 0.8
            val_ratio = 0.1
            train_size = int(train_ratio * self.size)
            val_size = int(val_ratio * self.size)
            test_size = self.size - train_size - val_size

            assert test_size >= 0

            self.train_dataset, self.val_dataset, self.test_dataset = \
                random_split(self.dataset, [train_size, val_size, test_size])
        else:
            keys = self.mturk_dataset.keys

            self.train_dataset, self.val_dataset, self.test_dataset = \
                split_dataset_according_to_locations(keys, self.dataset)

        if self.weighted_sampling:
            class_weights = torch.zeros(self.num_classes)
            for x, y in self.train_dataset:
                class_weights[y.argmax()] += 1
            weights = []
            for x, y in self.train_dataset:
                weights.append(1 / (100 + class_weights[y.argmax()].item()))
            self.train_sampler = WeightedRandomSampler(weights, len(self.train_dataset))
        else:
            self.train_sampler = None

    def __len__(self):
        return self.size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=(self.train_sampler is None),
                          drop_last=True, pin_memory=True, num_workers=self.num_workers,
                          sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          drop_last=True, pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          drop_last=True, pin_memory=True, num_workers=self.num_workers)
