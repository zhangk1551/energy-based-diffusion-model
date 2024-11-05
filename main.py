#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, "../torchdriveenv")

import os
import random
import argparse
import logging
from omegaconf import OmegaConf

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from edm.datasets import EDMDataModule
from edm.pl_modules import EDMModule
from edm.configs import EDMConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--sweep_id', type=str, default=None)
parser.add_argument('--wandb_mode', type=str, default="offline")
parser.add_argument('--conf', type=str,
                    default="conf/torchdriveenv_policy.yml")


# os.environ["WANDB_API_KEY"] = ""
PROJECT = "EDM"

def main(config=None):
    wandb_logger = WandbLogger(project=PROJECT, log_model=True)

    seed = config.seed
    if seed is None:
        seed = random.randint(0, 100000)
    pl.seed_everything(seed, workers=True)

    monitor = config.trainer.monitor
    monitor_mode = config.trainer.monitor_mode
#    wandb.define_metric(monitor, summary=monitor_mode)

    data_module = EDMDataModule(config)
    data_module.prepare_datasets()

    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=monitor_mode, save_top_k=2,
                                          filename="{epoch:03d}-{train_loss:.2f}")
#    early_stop_callback = EarlyStopping(monitor=monitor,
#                                        mode=monitor_mode,
#                                        patience=config.trainer.early_stopping_patience)

#                         strategy=config.trainer.strategy,

    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs,
                         fast_dev_run=config.trainer.fast_dev_run,
                         overfit_batches=config.trainer.overfit_batches,
                         accelerator='gpu',
                         devices=config.trainer.devices,
                         num_nodes=config.trainer.num_nodes,
                         logger=wandb_logger,
                         log_every_n_steps=5,
                         gradient_clip_val=0.5,
                         callbacks=[checkpoint_callback] if config.trainer.overfit_batches == 0
                         else None)

    edm_module = EDMModule(config, x_dim=data_module.x_dim, s_dim=data_module.s_dim, eval_data=data_module.eval_data)

    trainer.fit(edm_module, datamodule=data_module)

#    if not config.trainer.fast_dev_run:
#        wandb.save(checkpoint_callback.best_model_path)
#        trainer.test(ckpt_path="best", datamodule=data_module)


def merge_sweep_config(config, sweep_config):
    parameters = dict(sweep_config).keys()
    groups = dict(config).keys()
    for parameter in parameters:
        for group in groups:
            if parameter in config[group]:
                config[group][parameter] = sweep_config[parameter]
    return config


def cli_main():
    default_config = OmegaConf.structured(EDMConfig)

    if args.conf is not None:
        yaml_config = OmegaConf.load(args.conf)
        config = OmegaConf.merge(default_config, yaml_config)
    else:
        config = default_config

    if args.sweep_id is not None:
        wandb.init(project=PROJECT)
        sweep_config = wandb.config
        config = merge_sweep_config(config, sweep_config)

#    if args.data_dir is not None:
#        config["data"]["data_dir"] = args.data_dir

    if (config["trainer"]["devices"] == 1) and (config["trainer"]["num_nodes"] == 1):
        config["trainer"]["strategy"] = None
    else:
        config["trainer"]["strategy"] = "ddp"

    logger.info("config")
    logger.info(config)
    main(config)


if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb_mode

    if args.sweep_id is not None:
        wandb.agent(args.sweep_id, function=cli_main, project=PROJECT, count=1)
    else:
        cli_main()
