import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet(nn.Module):
    """
    Resnet score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(self,
                 n_steps,
                 n_layers,
                 x_dim,
                 h_dim,
                 emb_dim,
                 widen=2,
                 emb_type='learned'):
        # assert emb_type in ('learned', 'sinusoidal')
        super().__init__()
        self._n_layers = n_layers
        self._n_steps = n_steps
        self._x_dim = x_dim
        self._h_dim = h_dim
        self._emb_dim = emb_dim
        self._widen = widen
        self._emb_type = emb_type

        self.layer_x = nn.Linear(self._x_dim, self._h_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim, h_dim * widen),
                nn.Linear(emb_dim, h_dim * widen),
                nn.Linear(h_dim * widen, h_dim * widen),
                nn.Linear(h_dim * widen, h_dim)
            )
            for _ in range(n_layers)

        ])

        self.last_layer = nn.Linear(self._h_dim, self._x_dim)

        for i in range(self._n_layers):
            nn.init.zeros_(self.layers[i][-1].weight)
        nn.init.zeros_(self.last_layer.weight)

        self.time_emb = nn.Embedding(self._n_steps, self._emb_dim)

    def forward(self, x, t):

        x = torch.atleast_2d(x)
        t = torch.atleast_1d(t)

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


class EBMDiffusionModel(nn.Module):
    """
    EBM parameterization on top of score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def neg_logp_unnorm(self, x, t):
        score = self.net(x, t)
        return ((score - x) ** 2).sum(-1)

    def __call__(self, x, t):
        neg_logp_unnorm = lambda _x: self.neg_logp_unnorm(_x, t).sum()
        return torch.func.grad(neg_logp_unnorm)(x)
