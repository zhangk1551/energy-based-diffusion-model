import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CNN(pl.LightningModule):
    def __init__(self, input_shape, output_size=128):
        super().__init__()

        downsample_kernel_size = math.ceil(input_shape[-1] / 256)
        self.pool0 = nn.AvgPool2d(downsample_kernel_size)

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.pool1 = nn.MaxPool2d(2)

        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, output_size)

    def _get_conv_output(self, shape):
        input = torch.autograd.Variable(torch.rand(1, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool1(F.relu(self.conv4(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ScoreModel(nn.Module):
    """
    Resnet score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(self, config, n_steps, x_dim, s_dim):
        # assert emb_type in ('learned', 'sinusoidal')
        super().__init__()
        n_layers = config.n_layers
        h_dim = config.h_dim
        emb_dim = config.emb_dim
        widen = config.widen
        self.image_encoder = None

        if s_dim is not None:
            if type(s_dim) is int:
                self.layer_x = nn.Linear(x_dim + s_dim, h_dim)
            else:
                s_out_dim = 128
                self.image_encoder = CNN(input_shape=s_dim, output_size=s_out_dim)
                self.layer_x = nn.Linear(x_dim + s_out_dim, h_dim)
        else:
            self.layer_x = nn.Linear(x_dim, h_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim, h_dim * widen),
                nn.Linear(emb_dim, h_dim * widen),
                nn.Linear(h_dim * widen, h_dim * widen),
                nn.Linear(h_dim * widen, h_dim)
            )
            for _ in range(n_layers)
        ])

        self.last_layer = nn.Linear(h_dim, x_dim)

        for i in range(n_layers):
            nn.init.zeros_(self.layers[i][-1].weight)
        nn.init.zeros_(self.last_layer.weight)

        self.time_emb = nn.Embedding(n_steps, emb_dim)

    def forward(self, x, t, s):

        x = torch.atleast_2d(x)
        t = torch.atleast_1d(t)

        if s is not None:
            s = torch.atleast_2d(s)
            if self.image_encoder is not None:
                s = self.image_encoder(s)
            x = torch.cat([x, s], dim=-1)

        emb = self.time_emb(t)

        x = x.float()
        x = self.layer_x(x)

        for layer in self.layers:
            h = F.layer_norm(x, normalized_shape=x.size()[1:])
            h = F.silu(h)
            h = layer[0](h)
            h = h + layer[1](emb)
            h = F.silu(h)
            h = layer[2](h)
            x = x + layer[3](h)

        x = self.last_layer(x)

        return x


class EnergyModel(nn.Module):
    """
    EBM parameterization on top of score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def neg_logp_unnorm(self, x, t, s):
        score = self.net(x, t, s)
        return ((score - x) ** 2).sum(-1)

    def __call__(self, x, t, s):
        neg_logp_unnorm = lambda _x: self.neg_logp_unnorm(_x, t, s).sum()
        return torch.func.grad(neg_logp_unnorm)(x)



# class ConditionalResnet(nn.Module):
#     """
#     Resnet score model.
# 
#     Adds embedding for each scale after each linear layer.
#     """
# 
#     def __init__(self,
#                  n_steps,
#                  n_layers,
#                  x_dim,
#                  s_dim,
#                  h_dim,
#                  emb_dim,
#                  widen=2,
#                  emb_type='learned'):
#         # assert emb_type in ('learned', 'sinusoidal')
#         super().__init__()
#         self._n_layers = n_layers
#         self._n_steps = n_steps
#         self._x_dim = x_dim
#         self._s_dim = s_dim
#         self._h_dim = h_dim
#         self._emb_dim = emb_dim
#         self._widen = widen
#         self._emb_type = emb_type
# 
#         self.layer_x = nn.Linear(self._x_dim + self._s_dim, self._h_dim)
# 
#         self.layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(h_dim, h_dim * widen),
#                 nn.Linear(emb_dim, h_dim * widen),
#                 nn.Linear(h_dim * widen, h_dim * widen),
#                 nn.Linear(h_dim * widen, h_dim)
#             )
#             for _ in range(n_layers)
# 
#         ])
# 
#         self.last_layer = nn.Linear(self._h_dim, self._x_dim)
# 
#         for i in range(self._n_layers):
#             nn.init.zeros_(self.layers[i][-1].weight)
#         nn.init.zeros_(self.last_layer.weight)
# 
#         self.time_emb = nn.Embedding(self._n_steps, self._emb_dim)
# 
#     def forward(self, x, t, s):
# 
#         x = torch.atleast_2d(x)
#         s = torch.atleast_2d(s)
#         t = torch.atleast_1d(t)
# 
#         emb = self.time_emb(t)
# 
#         x = torch.cat([x, s], dim=-1)
# 
#         x = x.float()
#         x = self.layer_x(x)
# 
#         for layer in self.layers:
#             h = F.layer_norm(x, normalized_shape=x.size()[1:])
#             h = F.silu(h)
#             h = layer[0](h)
#             h = h + layer[1](emb)
#             h = F.silu(h)
#             h = layer[2](h)
#             x = x + layer[3](h)
# 
#         x = self.last_layer(x)
# 
#         return x



# class ConditionalEBMDiffusionModel(nn.Module):
#     """
#     EBM parameterization on top of score model.
# 
#     Adds embedding for each scale after each linear layer.
#     """
# 
#     def __init__(self, net):
#         super().__init__()
#         self.net = net
# 
#     def neg_logp_unnorm(self, x, t, s):
#         score = self.net(x, t, s)
#         return ((score - x) ** 2).sum(-1)
# 
#     def __call__(self, x, t, s):
#         neg_logp_unnorm = lambda _x: self.neg_logp_unnorm(_x, t, s).sum()
#         return torch.func.grad(neg_logp_unnorm)(x)
