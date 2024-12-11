import os
import torch
import random
import pytorch_lightning as pl
import numpy as np

from PIL import Image

from edm.diffusion import Diffusion
from edm.models import EnergyModel, SimpleResNet, Unet
from edm.utils import plot_samples, plot_hist, plot_energy, get_episode_data, to_gif


class EDMModule(pl.LightningModule):
    def __init__(self, config, x_dim, s_dim, eval_datasets=None, x_mean=None, x_std=None, x_transform=None):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.n_eval_samples = config.diffusion.n_eval_samples

        if config.model.image_encoder_x_path is not None:
            image_encoder_x = EDMModule.load_from_checkpoint(config.model.image_encoder_x_path).net.image_encoder_s
            x_dim = image_encoder_x.fc2.bias.shape[-1]
        else:
            image_encoder_x = None

#        self.net = SimpleResNet(config=config.model, n_steps=config.diffusion.n_steps, x_dim=x_dim, s_dim=s_dim, image_encoder_x=image_encoder_x)
        self.net = Unet()
        self.ebm_net = EnergyModel(self.net, image_encoder_x=image_encoder_x)
        self.diffusion = Diffusion(config.diffusion, self.ebm_net, dim=x_dim)

        self.eval_datasets = eval_datasets
        self.x_mean = x_mean
        self.x_std = x_std
        self.x_transform = x_transform


    def training_step(self, batch, batch_idx):
        x, s = batch
        loss = self.diffusion.loss(x, s)
        loss = loss.mean()
        self.log_dict({"train_loss": loss})
        return loss


    def on_train_epoch_end(self):
        self.diffusion.eval()
        self.ebm_net.eval()
        self.net.eval()
        with torch.no_grad():
            if self.config.diffusion_keys == ["obs_birdview"]:
                self.evaluate_obs_samples()
                self.evaluate_obs_distribution()
            if self.config.condition_keys is None:
                self.diffusion.train()
                self.ebm_net.train()
                self.net.train()
                return
            if self.current_epoch % 5 == 0:
                self.evaluate_model()
        self.diffusion.train()
        self.ebm_net.train()
        self.net.train()


    def pick_evaluate_episode(self):
        data_dir = "/home/kezhang/work/fall_2024/energy-based-diffusion-model/data/selected_more_waypoints_val"
        file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        while True:
            file_path = random.choice(file_paths)
#            file_path = "/home/kezhang/work/fall_2024/energy-based-diffusion-model/data/waypoint_graph_no_rendering_test/episode_9_95910.pkl"
            obs_images, recurrent_states, actions = get_episode_data(file_path)
            if len(actions) > 0:
                return obs_images, recurrent_states, actions


    def evaluate_obs_distribution(self):
        images = []
        means = []
        for dataset in self.eval_datasets:
            energy_list = []
            for obs in dataset:
                if self.x_transform is not None:
                    obs = self.x_transform(obs)
                energy = self.diffusion.p_energy(obs.unsqueeze(0).to(self.device), t=torch.Tensor([0]).int().to(self.device), s=None, clip=1.0)
                energy_list.append(energy.item())
            images.append(plot_hist(energy_list))
            means.append(sum(energy_list) / len(energy_list))
        self.logger.log_image(key="state distributions", images=images, caption=[f"bad policy validation mean energy:{means[0]}",
                                                                                 f"in-distribution validation mean energy:{means[1]}",
                                                                                 f"irrelevant_shuffled validation mean energy:{means[2]}",
                                                                                 f"irrelevant_cat validation mean energy:{means[3]}"])

    def evaluate_obs_samples(self):
        obs_samples = self.diffusion.sample(s=None, n=self.n_eval_samples, shape=(3, 64, 64), clip=1.0)
#        print("obs_samples")
#        print(obs_samples)
        image_tensors = [torch.round((sample * self.x_std[:, None, None].cuda() + self.x_mean[:, None, None].cuda()) * 255.0) for sample in obs_samples]
        obs_images = [Image.fromarray(img.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0), 'RGB') for img in image_tensors]
        self.logger.log_image(key="state samples", images=obs_images)


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
        if optim_method == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        if optim_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        return optimizer
