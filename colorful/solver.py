import torch
from torch.nn import functional
from colorful import cielab, encoders, CELoss
from util import module


class ColorfulSolver(module.Solver):
    def __init__(self, network, config):
        super().__init__(network, config)

        # Loss function and re-balancing
        self.cielab = cielab.LABBins()
        self.cielab_gamut = cielab.ABGamut()
        self.rebalace = self.derived_config['lambda'] > 0

        if self.rebalace:
            self.prior = torch.from_numpy(self.cielab_gamut.priors).type(self.dtype).to(self.device)
            self.l_factor = self.derived_config['lambda']
            uniform = torch.ones_like(self.prior)
            uniform /= len(self.prior)

            self.weights = 1 / ((1 - self.l_factor) * self.prior + self.l_factor * uniform)
            self.weights /= torch.sum(self.prior * self.weights)

            self.loss = CELoss.MultinomialCrossEntropyLoss(self.weights)
        else:
            self.loss = CELoss.MultinomialCrossEntropyLoss()

        # Encoder
        self.encoder = encoders.SoftEncoder(self.cielab, device=self.device)

    def calculate_loss(self, batch):
        # Fetch input for the network
        x = batch[:, :1, :, :].to(self.device)
        # Forward pass
        predicted = self.network(x)
        # Fetch output and resize
        actual = functional.interpolate(batch[:, 1:, :, :], size=predicted.shape[2:]).to(self.device)
        # Encode the outputs
        labels = self.encoder(actual)

        # Calculate loss
        return self.loss(predicted, labels)
