import numpy as np
import util
from functools import partial

class ABGamut:
    POINTS_PATH = "colorful/resources/points.npy"
    PRIORS_PATH = "colorful/resources/priors.npy"

    def __init__(self, dtype=np.float32):
        self.points = np.load(self.POINTS_PATH).astype(dtype)
        self.priors = np.load(self.PRIORS_PATH).astype(dtype)

        assert self.points.shape[0] == self.priors.shape[0]

        self.bins = self.points.shape[0]

    def __len__(self):
        return self.bins


class LABBins:
    L_MEAN = 50
    BINSIZE = 10

    def __init__(self, gamut=None):
        self.gamut = gamut if gamut is not None else ABGamut()

        a = np.arange(
            -110 - self.BINSIZE // 2,
             110 + self.BINSIZE // 2,
            self.BINSIZE,
            dtype=np.float32)
        b = np.arange(
            -110 - self.BINSIZE // 2,
             110 + self.BINSIZE // 2,
            self.BINSIZE,
            dtype=np.float32)

        b_, a_ = np.meshgrid(a, b)
        ab = np.dstack((a_, b_))

        self.ab_gamut_mask = np.full(ab.shape[:-1], False, dtype=bool)

        a = np.digitize(self.gamut.points[:, 0], a) - 1
        b = np.digitize(self.gamut.points[:, 1], b) - 1

        for a_, b_ in zip(a, b):
            self.ab_gamut_mask[a_, b_] = True

        self.ab_to_q = np.full(self.ab_gamut_mask.shape, -1, dtype=np.int64)
        self.ab_to_q[self.ab_gamut_mask] = np.arange(self.gamut.bins)

        self.q_to_ab = ab[self.ab_gamut_mask] + self.BINSIZE / 2

    def ab2q(self, ab):
        ab_discrete = ((ab + 110) / self.BINSIZE).astype(int)

        a, b = np.hsplit(ab_discrete.reshape(-1, 2), 2)

        return self.ab_to_q[a, b].reshape(*ab.shape[:2])

    def q2ab(self, q):
        return self.q_to_ab[q]
