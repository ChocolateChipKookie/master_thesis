import math
import torch


class SoftEncoder:
    def __init__(self, cielab, neighbours=5, sigma=5.0, device="cpu"):
        self.bins = cielab.gamut.bins
        self.q_to_ab = torch.from_numpy(cielab.q_to_ab).to(device)

        self.neighbours = neighbours
        self.sigma = sigma
        self.norm = 1 / (2 * math.pi * self.sigma)

    def __call__(self, ab):
        n, _, h, w = ab.shape
        m = n * h * w

        # find nearest neighbours
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        cdist = torch.cdist(self.q_to_ab, ab_.t())
        nns = cdist.argsort(dim=0)[:self.neighbours, :]

        # gaussian weighting
        nn_gauss = ab.new_zeros(self.neighbours, m)
        for i in range(self.neighbours):
            tmp = torch.exp(-torch.sum((self.q_to_ab[nns[i, :], :].t() - ab_) ** 2, dim=0) / (2 * self.sigma ** 2))
            nn_gauss[i, :] = self.norm * tmp

        nn_gauss /= nn_gauss.sum(dim=0, keepdim=True)

        # expand
        q = ab.new_zeros(self.bins, m)
        q[nns, torch.arange(m).repeat(self.neighbours, 1)] = nn_gauss

        return q.reshape(self.bins, n, h, w).permute(1, 0, 2, 3)

    def to(self, device):
        self.q_to_ab = self.q_to_ab.to(device)


class HardEncoder:
    def __init__(self, cielab, device="cpu"):
        self.bins = cielab.gamut.bins
        self.q_to_ab = torch.from_numpy(cielab.q_to_ab).to(device)

    def __call__(self, ab):
        n, _, h, w = ab.shape
        m = n * h * w

        # find nearest neighbours
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        classes = torch.cdist(self.q_to_ab, ab_.t()).argmin(dim=0)
        classes = classes.reshape(n, h, w)
        return classes

    def to(self, device):
        self.q_to_ab = self.q_to_ab.to(device)
