from torch.nn import functional
import torch
from util import module, util
from skimage import color
from torchvision.transforms import functional
from colorful import cielab, encoders


class GlobalHints:
    def __init__(self, soft=True):
        self.cielab = cielab.LABBins()

        if soft:
            self.encoder = encoders.SoftEncoder(self.cielab)
            self.encode = self.soft_encoded
        else:
            self.encoder = encoders.HardEncoder(self.cielab)
            self.encode = self.hard_encoded

    def soft_encoded(self, batch, random_saturation=False, random_histogram=False):
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
                histogram = torch.sum(encoded, dim=(0, 2, 3))
                hints[i, 3:] = histogram / torch.sum(histogram)
        return hints

    def hard_encoded(self, batch, random_saturation=False, random_histogram=False):
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

    def get_global_hints(self, batch, random_saturation=False, random_histogram=False):
        return self.encode(batch, random_saturation, random_histogram)

    def __call__(self, batch, random_saturation=False, random_histogram=False):
        return self.get_global_hints(batch, random_saturation, random_histogram)


class Solver(module.Solver):
    def __init__(self, network, config):
        super(Solver, self).__init__(network, config)
        self.loss_func = util.factory(self.derived_config['loss'])
        self.global_hints = GlobalHints()
        self.random_hints = self.derived_config["random_hints"]

        # Re-create optimizer, optimize only layers that have been added
        optimizer_data = self.solver_config['optimizer']
        optimizer_class = util.import_attr(optimizer_data['class'])
        parameters = []
        parameters.extend(self.network.conv8up.parameters())
        parameters.extend(self.network.conv3_8short.parameters())
        parameters.extend(self.network.conv8.parameters())

        parameters.extend(self.network.conv9up.parameters())
        parameters.extend(self.network.conv2_9short.parameters())
        parameters.extend(self.network.conv9.parameters())

        parameters.extend(self.network.conv10up.parameters())
        parameters.extend(self.network.conv1_10short.parameters())
        parameters.extend(self.network.conv10.parameters())

        parameters.extend(self.network.global_hints.parameters())
        parameters.extend(self.network.out.parameters())

        if self.derived_config["optimize_bottleneck"]:
            parameters.extend(self.network.conv4.parameters())
            parameters.extend(self.network.conv5.parameters())
            parameters.extend(self.network.conv6.parameters())
            parameters.extend(self.network.conv7.parameters())

        self.optimizer = optimizer_class(parameters, **optimizer_data['args'])

        if "base_optimizer" in self.derived_config:
            optimizer_data = self.derived_config['base_optimizer']
            optimizer_class = util.import_attr(optimizer_data['class'])
            parameters = []

            parameters.extend(self.network.conv1.parameters())
            parameters.extend(self.network.conv1_down.parameters())
            parameters.extend(self.network.conv2.parameters())
            parameters.extend(self.network.conv2_down.parameters())
            parameters.extend(self.network.conv3.parameters())
            parameters.extend(self.network.conv3_down.parameters())
            parameters.extend(self.network.conv4.parameters())
            parameters.extend(self.network.conv5.parameters())
            parameters.extend(self.network.conv6.parameters())
            parameters.extend(self.network.conv7.parameters())

            optimizers = [self.optimizer, optimizer_class(parameters, **optimizer_data['args'])]
            self.optimizer = module.MultiOptimizer(optimizers)

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