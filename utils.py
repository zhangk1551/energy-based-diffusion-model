import torch
import numpy as np

from matplotlib import pyplot as plt

def plot_samples(x):
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

def dist_show_2d(fn, xr, yr):
    nticks = 100
    x, y = np.meshgrid(np.linspace(xr[0], xr[1], nticks), np.linspace(yr[0], yr[1], nticks))
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    coord = torch.stack([x, y], axis=-1).reshape((-1, 2))
    heatmap = fn(coord).reshape((nticks, nticks)).detach().numpy()
    plt.imshow(heatmap)

def extract(a: torch.Tensor,
            t: torch.Tensor,
            x_shape) -> torch.Tensor:
    """Get coefficients at given timesteps and reshape to [batch_size, 1, ...]."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = a[t]

    assert out.shape[0] == bs

    return out.reshape([bs] + (len(x_shape) - 1) * [1])

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)
