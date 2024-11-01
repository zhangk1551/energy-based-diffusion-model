import torch
import pytorch_lightning as pl

from diffusion import Diffusion
from models import EnergyModel, ScoreModel


class EDMModule(pl.LightningModule):
    def __init__(self, config, x_dim, s_dim):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        self.net = ScoreModel(config=config.model, n_steps=config.diffusion.n_steps, x_dim=x_dim, s_dim=s_dim)
        self.ebm_net = EnergyModel(self.net)
        self.diffusion = Diffusion(config.diffusion)

#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#        if config.is_conditional:
#            self.net = ConditionalResnet(n_steps=config.n_steps,
#                                         n_layers=4,
#                                         x_dim=config.data_dim,
#                                         h_dim=128,
#                                         emb_dim=32)
#            self.ebm_net = ConditionalEBMDiffusionModel(self.net)
#        else:
#            self.net = Resnet(n_steps=config.n_steps,
#                              n_layers=4,
#                              x_dim=config.data_dim,
#                              h_dim=128,
#                              emb_dim=32)
#            self.ebm_net = EnergyModel(self.net)
#
#        self.diffusion_model = DiffusionModel(config.data_dim,
#                                              config.n_steps,
#                                              self.ebm_net,
#                                              self.device,
#                                              var_type="beta_forward").to(self.device)

    def training_step(self, batch, batch_idx):
        x, s = batch
        loss = self.diffusion.loss(x, s)
        loss = loss.mean()
        self.log_dict({"train_loss": loss})
        return loss

#    def validation_step(self, batch, batch_idx):
#        self._shared_eval_step(batch, "val")
#
#    def test_step(self, batch, batch_idx):
#        self._shared_eval_step(batch, "test")
#
#    def predict_step(self, batch, batch_idx):
#        x, y = batch
#        y_hat = self.model(x).squeeze()
#        return y_hat
#
#    def _shared_eval_step(self, batch, stage):
#        x, y = batch
#        y_hat = self.model(x).squeeze()
#        loss = self.criterion(y_hat, self._format_y(y))
#        self.log_dict({f"{stage}_loss": loss})

#        metrics = self.calc_metrics(y_hat, y, self.config.module.task.value, self.config.module.num_classes, stage)
#        self.log_dict(metrics)
#
#    def calc_metrics(self, y_hat, y, task, num_classes, stage):
#
#        return {f"{stage}_accuracy": acc,
#                f"{stage}_precision": prec,
#                f"{stage}_recall": rec,
#                f"{stage}_f1": f1,
#                f"{stage}_mean_absolute_error": mae}

    def configure_optimizers(self):
        lr = self.config.trainer.lr
        optim_method = self.config.trainer.optim_method
        if optim_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if optim_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        return optimizer
