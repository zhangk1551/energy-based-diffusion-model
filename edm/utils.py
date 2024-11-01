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

