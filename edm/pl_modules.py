import os
import torch
import random
import pytorch_lightning as pl
import numpy as np

from PIL import Image

from edm.diffusion import Diffusion
from edm.models import EnergyModel, ScoreModel
from edm.utils import plot_samples, plot_energy, get_episode_data, to_gif


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


    def training_step(self, batch, batch_idx):
        x, s = batch
        loss = self.diffusion.loss(x, s)
        loss = loss.mean()
        self.log_dict({"train_loss": loss})
        return loss


    def on_train_epoch_end(self):
        if self.current_epoch % 20 == 0:
            self.evaluate_model()


    def pick_evaluate_episode(self):
        data_dir = "/home/kezhang/work/fall_2024/energy-based-diffusion-model/data/waypoint_graph_no_rendering_test"
        file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        while True:
            file_path = random.choice(file_paths)
            file_path = "/home/kezhang/work/fall_2024/energy-based-diffusion-model/data/waypoint_graph_no_rendering_test/episode_9_95910.pkl"
            obs_images, recurrent_states, actions = get_episode_data(file_path)
            if len(actions) > 0:
                return obs_images, recurrent_states, actions


    def evaluate_model(self):
        obs_birdviews, recurrent_states, actions = self.pick_evaluate_episode()
        obs_images = [Image.fromarray(img.cpu().numpy().astype(np.uint8).transpose(1, 2, 0), 'RGB') for img in obs_birdviews]
        x_samples = [self.diffusion.sample(s=obs_birdview.to(self.device), n=self.n_eval_samples).cpu().detach().numpy() for obs_birdview in obs_birdviews]
#        x_samples = [self.diffusion.sample(s=recurrent_state.to(self.device), n=self.n_eval_samples).cpu().detach().numpy() for recurrent_state in recurrent_states]
        sample_images = [plot_samples(x_sample) for x_sample in x_samples]

#        energy_images = [plot_energy(self.diffusion.p_energy, t=0, s=recurrent_state, device=self.device) for recurrent_state in recurrent_states]
        energy_images = [plot_energy(self.diffusion.p_energy, t=0, s=obs_birdview, device=self.device) for obs_birdview in obs_birdviews]

        self.logger.log_video(key="samples",
                              videos=[to_gif(obs_images, gif_name="obs.gif"),
                                      to_gif(sample_images, gif_name="sample.gif"),
                                      to_gif(energy_images, gif_name="energy.gif")],
                              caption=["observation", "action samples", "action_energy"],
                              format=["gif"] * 3)


    def configure_optimizers(self):
        lr = self.config.trainer.lr
        optim_method = self.config.trainer.optim_method
        if optim_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if optim_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        return optimizer
