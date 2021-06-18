from torch.nn import functional
import torch
from util import module, util
from skimage import color
from torchvision.transforms import functional
from colorful import cielab, encoders


class GlobalHints:
    def __init__(self):
        self.cielab = cielab.LABBins()
        self.encoder = encoders.HardEncoder(self.cielab)

    def get_global_hints(self, batch, random_saturation=False, random_histogram=False):
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

    def __call__(self, batch, random_saturation=False, random_histogram=False):
        return self.get_global_hints(batch, random_saturation, random_histogram)

class Solver(module.Solver):
    def __init__(self, network, config):
        super(Solver, self).__init__(network, config)
        self.loss_func = util.factory(self.derived_config['loss'])
        self.global_hints = GlobalHints()
        self.random_hints = self.derived_config["random_hints"]

        if not self.derived_config["fine_tune_existing"]:
            # Re-create optimizer, optimize only layers that have been added
            optimizer_data = self.solver_config['optimizer']
            optimizer_class = util.import_attr(optimizer_data['class'])
            parameters = []
            parameters.extend(self.network.conv3_8short.parameters())

            parameters.extend(self.network.conv9up.parameters())
            parameters.extend(self.network.conv9.parameters())
            parameters.extend(self.network.conv9.parameters())

            parameters.extend(self.network.conv10up.parameters())
            parameters.extend(self.network.conv1_10short.parameters())
            parameters.extend(self.network.conv10.parameters())

            parameters.extend(self.network.global_hints.parameters())
            parameters.extend(self.network.out.parameters())

            self.optimizer = optimizer_class(parameters, **optimizer_data['args'])

    def get_global(self, batch, random_saturation=False, random_histogram=False):
        return self.global_hints(batch, random_saturation, random_histogram)

    def calculate_loss(self, batch):
        return self.loss(batch, self.random_hints)

    def loss(self, batch, random_hints):
        # Fetch input for the network
        global_hints = self.global_hints(batch, random_hints, random_hints).to(self.device)
        x = batch[:, :1, :, :].to(self.device)
        # Forward pass
        predicted = self.network(x, global_hints)
        # Fetch output and resize
        actual = self.network.normalize_ab(batch[:, 1:, :, :]).to(self.device)
        # Calculate loss
        return self.loss_func(predicted, actual)