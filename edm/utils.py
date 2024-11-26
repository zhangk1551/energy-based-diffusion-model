import cv2
import io
import torch
import pickle
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt


def to_image(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def to_video(pil_images, video_name = "video.mp4", fps=10):

    frames = [np.array(img) for img in pil_images]

    height, width, layers = frames[0].shape
    size = (width, height)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_name,
                               fourcc, fps, size)

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    video_writer.release()
    return video_name


def to_gif(pil_images, gif_name=None, duration=100):
    save_to = gif_name if gif_name is not None else io.BytesIO()
    pil_images[0].save(save_to,
                       format="GIF",
                       save_all=True,
                       append_images=pil_images[1:],
                       duration=duration,
                       loop=0)
    return save_to


def plot_samples(x):
    plt.scatter(x[:, 1], x[:, 0])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    return to_image(plt)


def get_value(key, step_data):
    if key == "obs_birdview":
        return torch.Tensor(step_data.obs_birdview).squeeze()
    if key == "recurrent_state":
        return torch.Tensor(step_data.recurrent_states[0]).squeeze()
    if key == "action":
        return step_data.ego_action.squeeze()


def get_episode_data(episode_file_path):
    obs_birdviews = []
    actions = []
    recurrent_states = []
    with open(episode_file_path, "rb") as f:
        episode_data = pickle.load(f)
        for step_data in episode_data.step_data:
            obs_birdviews.append(
                get_value("obs_birdview", step_data))
            action = get_value("action", step_data)
            actions.append(action)
            recurrent_states.append(get_value("recurrent_state", step_data))

#    return [Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0), 'RGB') for img in obs_birdviews], \
#            recurrent_states, \
#            actions
    return  obs_birdviews, recurrent_states, actions


def plot_energy(fn, t, s, device):
    nticks = 50
    x, y = np.meshgrid(np.linspace(-2, 2, nticks), np.linspace(-2, 2, nticks))
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    coord = torch.stack([x, y], axis=-1).reshape((-1, 2)).to(device)
    t = torch.Tensor([t]).int().expand((coord.shape[0], )).to(device)
    s = s.unsqueeze(0).expand((coord.shape[0], *s.shape)).to(device)

    heatmap = fn(x=coord, t=t, s=s).reshape(
        (nticks, nticks)).detach().cpu().transpose(0, 1).numpy()
    plt.imshow(heatmap)

    return to_image(plt)
