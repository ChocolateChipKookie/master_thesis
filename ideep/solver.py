from torch.nn import functional
import torch
from util import module, util


class Solver(module.Solver):
    def __init__(self, network, config):
        super(Solver, self).__init__(network, config)
        self.loss = util.factory(self.derived_config['loss'])

    def calculate_loss(self, batch):
        # Fetch input for the network
        x = batch[:, :1, :, :].to(self.device)
        # Forward pass
        predicted = self.network(x)
        # Fetch output and resize
        actual = self.network.normalize_ab(batch[:, 1:, :, :]).to(self.device)

        # Calculate loss
        return self.loss(predicted, actual)