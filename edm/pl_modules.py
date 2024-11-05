import io
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from PIL import Image

from edm.diffusion import Diffusion
from edm.models import EnergyModel, ScoreModel
from edm.utils import plot_samples, plot_energy


class EDMModule(pl.LightningModule):
    def __init__(self, config, x_dim, s_dim, eval_data=None):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.n_eval_samples = config.diffusion.n_eval_samples

        self.net = ScoreModel(config=config.model, n_steps=config.diffusion.n_steps, x_dim=x_dim, s_dim=s_dim)
        self.ebm_net = EnergyModel(self.net)
        self.diffusion = Diffusion(config.diffusion, self.ebm_net, dim=x_dim)

        self.eval_data = eval_data

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


    def on_train_epoch_end(self):
        if self.current_epoch % 100 == 0:
            self.evaluate_model()


    def evaluate_model(self):
#        plt.figure(figsize=(6, 4))
        x_samp = self.diffusion.sample(s=self.eval_data["recurrent_state"].to(self.device), n=self.n_eval_samples).cpu().detach().numpy()
        sample_image = plot_samples(x_samp)

        energy_image = plot_energy(self.diffusion.p_energy, t=0, s=self.eval_data["recurrent_state"], device=self.device)

        self.logger.log_image(key="samples", images=[self.eval_data["obs"], sample_image, energy_image], caption=["observation", "action samples", "action_energy"])
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
