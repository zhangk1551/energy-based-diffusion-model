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

        self.fc1 = nn.Linear(n_sizes, 256)
        self.fc2 = nn.Linear(256, output_size)

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


class SimpleResNet(nn.Module):
    """
    Resnet score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(self, config, n_steps, x_dim, s_dim, image_encoder_x=None):
        # assert emb_type in ('learned', 'sinusoidal')
        super().__init__()
        n_layers = config.n_layers
        h_dim = config.h_dim
        emb_dim = config.emb_dim
        widen = config.widen
        self.image_encoder_x = image_encoder_x
        self.image_encoder_s = None

        if s_dim is not None:
            if type(s_dim) is int:
                self.layer_x = nn.Linear(x_dim + s_dim, h_dim)
            else:
                s_out_dim = 128
                self.image_encoder_s = CNN(input_shape=s_dim, output_size=s_out_dim)
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

        x = x.float()

        if self.image_encoder_x is not None:
            x = self.image_encoder_x(x)

        if s.numel() > 0:
            s = torch.atleast_2d(s)
            if self.image_encoder_s is not None:
                s = self.image_encoder_s(s)
            x = torch.cat([x, s], dim=-1)

        emb = self.time_emb(t)

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

class ResNetBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout_rate=0.1):
        super(ResNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out += identity
        return out


class LN_Resnet(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16, hidden_size=256, dropout_rate=0.1):
        super(LN_Resnet, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = state_dim + action_dim + t_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.resnet_block1 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.resnet_block2 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.resnet_block3 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.input_layer(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.output_layer(x)
        return x


class Unet(nn.Module):
    def __init__(self,
        n_channel: int=3,
        D: int = 64,
        device: torch.device = torch.device("cuda"),
        ) -> None:
        super(Unet, self).__init__()
        self.device = device

        self.D = D

        self.freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=D, dtype=torch.float32) / D
        )

        blk = lambda ic, oc: nn.Sequential(
            nn.GroupNorm(32, num_channels=ic),
            nn.SiLU(),
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GroupNorm(32, num_channels=oc),
            nn.SiLU(),
            nn.Conv2d(oc, oc, 3, padding=1),
        )

        self.down = nn.Sequential(
            *[
                nn.Conv2d(n_channel, D, 3, padding=1),
                blk(D, D),
                blk(D, 2 * D),
                blk(2 * D, 2 * D),
            ]
        )

        self.time_downs = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, 2 * D),
            nn.Linear(2 * D, 2 * D),
        )

        self.mid = blk(2 * D, 2 * D)

        self.up = nn.Sequential(
            *[
                blk(2 * D, 2 * D),
                blk(2 * 2 * D, D),
                blk(D, D),
                nn.Conv2d(2 * D, 2 * D, 3, padding=1),
            ]
        )
        self.last = nn.Conv2d(2 * D + n_channel, n_channel, 3, padding=1)

    def forward(self, x, t, s=None) -> torch.Tensor:
        # time embedding
        x = x.float()

        args = t.float().unsqueeze(-1).expand(-1, self.D) * self.freqs[None].to(t.device)
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(x.device)

        x_ori = x

        # perform F(x, t)
        hs = []
        for idx, layer in enumerate(self.down):
            if idx % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
                x = F.interpolate(x, scale_factor=0.5)
                hs.append(x)

            x = x + self.time_downs[idx](t_emb)[:, :, None, None]

        x = self.mid(x)

        for idx, layer in enumerate(self.up):
            if idx % 2 == 0:
                x = layer(x) + x
            else:
                x = torch.cat([x, hs.pop()], dim=1)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = layer(x)

        x = self.last(torch.cat([x, x_ori], dim=1))

        return x


class EnergyModel(nn.Module):
    """
    EBM parameterization on top of score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(self, net, image_encoder_x=None):
        super().__init__()
        self.net = net
        self.image_encoder_x = image_encoder_x

    def neg_logp_unnorm(self, x, t, s):
        score = self.net(x, t, s)
        x = x.float()
        if self.image_encoder_x is not None:
            x = self.image_encoder_x(x)
        return ((score - x) ** 2).sum(dim=tuple(range(1,len(x.shape))))

    def __call__(self, x, t, s):
        neg_logp_unnorm = lambda _x: self.neg_logp_unnorm(_x, t, s).sum()
        return torch.func.grad(neg_logp_unnorm)(x)
