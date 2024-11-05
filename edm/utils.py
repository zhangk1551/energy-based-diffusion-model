import io
import torch
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt


def to_image(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def plot_samples(x):
    plt.scatter(x[:, 1], x[:, 0])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    return to_image(plt)


def plot_energy(fn, t, s, device):
    nticks = 100
    x, y = np.meshgrid(np.linspace(-2, 2, nticks), np.linspace(-2, 2, nticks))
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    coord = torch.stack([x, y], axis=-1).reshape((-1, 2)).to(device)
    t = torch.Tensor([t]).int().expand((coord.shape[0], )).to(device)
    s = s.unsqueeze(0).expand((coord.shape[0], -1)).to(device)

    heatmap = fn(x=coord, t=t, s=s).reshape((nticks, nticks)).detach().cpu().transpose(0, 1).numpy()
    plt.imshow(heatmap)

    return to_image(plt)
