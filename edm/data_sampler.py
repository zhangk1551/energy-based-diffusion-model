import torch


class DataSampler(object):
    def __init__(self, data, device):
        self.data = {}
        for key in data:
            self.data[key] = torch.from_numpy(data[key]).float()

        self.size = next(iter(data.values())).shape[0]
        self.device = device


    def sample(self, keys, batch_size, concat=False):
        ind = torch.randint(0, self.size, size=(batch_size,))
        samples = {}
        for key in keys:
          samples[key] = self.data[key][ind].to(self.device)

        if concat:
            return torch.cat(tuple(samples.values()), dim=-1)
        return samples
