import torch
from torch.nn import functional
from colorful import cielab, encoders, CELoss
from util import module


class UnetSolver(module.Solver):
    def __init__(self, network, config):
        super().__init__(network, config)

    def calculate_loss(self, batch):
        y = self.network.normalize(batch).to(self.device)
        # Fetch input for the network
        x = batch[:, :1, :, :].to(self.device)
        # Forward pass
        predicted = self.network(x)
        # Calculate loss
        return torch.cdist(predicted, y).mean()

