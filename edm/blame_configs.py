from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


@dataclass
class DataConfig:
    data_dir: str = "collected_collisions"
    label_file: Optional[str] = None
    mturk_file: Optional[str] = None
    batch_size: int = 2
    dataloader_num_workers: int = 0
    raw: bool = False
    ego_birdview_res: Optional[int] = None # if None, use the global birdview
    use_states: bool = False
    is_random_split: bool = False
    resolution: int = 256


@dataclass
class ModelConfig:
    name: str = "classifier"
    image_encoder: str = "cnn"
    layer_sizes: List[int] = field(default_factory=lambda: [128, 64, 32, 16])
    dropout_p: float = 0.0
    downsample_kernel_size: int = 4 # for crnn


class ModuleType(Enum):
    classification = 0
    regression = 1


class Task(Enum):
    binary = "binary"
    multiclass = "multiclass"
    multilabel = "multilabel"


@dataclass
class ModuleConfig:
    module_type: ModuleType = ModuleType.classification
    task: Task = Task.multiclass
    num_classes: int = 4
    predict_reason: bool = False
    weighted_sampling: bool = False


@dataclass
class TrainerConfig:
    monitor: str = "val_accuracy"
    monitor_mode: str = "max"
    max_epochs: int = 20
    fast_dev_run: bool = True
    overfit_batches: float = 0.0
    devices: int = 1
    num_nodes: int = 1
    strategy: Optional[str] = None
    lr: float = 0.0001
    optim_method: str = "SGD"
    lr_scheduler_factor: float = 0.9
    lr_scheduler_patience: int = 2
    early_stopping_patience: int = 5


@dataclass
class BlameConfig:
    data: DataConfig
    model: ModelConfig
    module: ModuleConfig
    trainer: TrainerConfig
