from torch.nn import functional
import torch
from util import module, util
from skimage import color
from torchvision.transforms import functional
from colorful import cielab, encoders


class Solver(module.Solver):
    def __init__(self, network, config):
        super(Solver, self).__init__(network, config)
        self.loss = util.factory(self.derived_config['loss'])
        # Loss function and re-balancing
        self.cielab = cielab.LABBins()
        self.encoder = encoders.HardEncoder(self.cielab)

    def get_global(self, batch, random_saturation=False, random_histogram=False):
        b, n, w, h = batch.shape
        hints = torch.zeros((b, 316))

        hints[:, :2] = 1
        if random_saturation:
            hints[:, 0] = torch.round(torch.rand(b))
        if random_histogram:
            hints[:, 1] = torch.round(torch.rand(b))

        for i in range(b):
            # Skip if both flags are false
            if hints[i, 0] == 0 and hints[i, 1] == 0:
                continue

            img = functional.resize(batch[i], [w//4, h//4])
            if hints[i, 0] > 0.5:
                rgb = color.lab2rgb(img.permute(1, 2, 0))
                hsv = color.rgb2hsv(rgb)
                saturation = hsv[:, :, 1].mean()
                hints[i, 2] = saturation.item()
            if hints[i, 1] > 0.5:
                # Calculate histogram
                ab = torch.unsqueeze(img[1:], 0)
                encoded = self.encoder(ab)
                bins = torch.bincount(encoded.view(-1), minlength=313)
                hints[i, 3:] = bins / torch.sum(bins)
        return hints

    def calculate_loss(self, batch):
        # Fetch input for the network
        global_hints = self.get_global(batch, True, True).to(self.device)
        x = batch[:, :1, :, :].to(self.device)
        # Forward pass
        predicted = self.network(x, global_hints)
        # Fetch output and resize
        actual = self.network.normalize_ab(batch[:, 1:, :, :]).to(self.device)

        # Calculate loss
        return self.loss(predicted, actual)