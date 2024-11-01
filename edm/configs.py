from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class DataConfig:
    train_data_dir: str
    val_data_dir: Optional[str] = None
    batch_size: int = 2
    dataloader_num_workers: int = 0
    constraints: Optional[Dict[str, List[str]]] = None


@dataclass
class ModelConfig:
    h_dim: int = 128
    emb_dim: int = 32
    n_layers: int = 4
    widen: int = 2
    emb_type: str = 'learned'


@dataclass
class DiffusionConfig:
    n_step: int = 100
    mc_loss: bool = True
    loss_type: str = 'simple'
    var_type: str = 'learned'
    samples_per_step: int = 1


@dataclass
class TrainerConfig:
    monitor: str = "train_loss"
    monitor_mode: str = "min"
    max_epochs: int = 20
    fast_dev_run: bool = True
    overfit_batches: float = 0.0
    devices: int = 1
    num_nodes: int = 1
    lr: float = 0.0001
    optim_method: str = "SGD"


@dataclass
class EDMConfig:
    task: str
    keys: List[str]
    condition_keys: Optional[List[str]] = None
    seed: Optional[int] = None
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    trainer: TrainerConfig
