import colorful.model
import util

import torch
from torch.nn import functional
import torchvision
from torch.utils.data import DataLoader
from colorful import cielab, encoders, CELoss

class Solver:
    def __init__(self, config):
        # Main options
        """
            device:
                - The device where the net is situated (cuda:0 or cpu)
            dtype:
                - Floating point datatype used for the net (torch.float16/32/64)
            lr:
                - Learning rate for the training process
            weight_decay:
                - Weight decay for the training process
            iterations:
                - Number of iterations to run the training
            batch_size:
                - Minibatch size for the training process
            data_path:
                - Training data path
            dataloader_workers:
                - Number of workers for the dataloader
            soft_encoding:
                - Use soft encoding
                - Bool value
        """
        # Main options
        self.device = torch.device(config['device'])
        self.dtype = config['dtype']

        # Network
        self.network = colorful.model.Colorful().type(self.dtype).to(self.device)

        # Net optimization
        self.learning_rate = config['lr']
        self.weight_decay = config['weight_decay']

        self.iterations = config['iterations']
        self.batch_size = config['batch_size']

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99)
        )

        # Loss function and re-balancing
        self.cielab = cielab.LABBins()
        self.cielab_gamut = cielab.ABGamut()
        self.rebalace = 'lambda' in config


        if 'lambda' in config:
            self.prior = torch.from_numpy(self.cielab_gamut.priors).type(self.dtype).to(self.device)
            self.l_factor = config['lambda']
            uniform = torch.ones_like(self.prior)
            uniform /= len(self.prior)

            self.weights = 1 / ((1 - self.l_factor) * self.prior + self.l_factor * uniform)
            self.weights /= torch.sum(self.prior * self.weights)

            self.loss = CELoss.CrossEntropyLoss(self.weights)
        else:
            self.loss = CELoss.CrossEntropyLoss()

        # Dataset and data loader
        self.transform = torchvision.transforms.Compose([
            util.ShortResize(256),
            torchvision.transforms.RandomCrop(256),
            util.Lab2rgb(),
            torchvision.transforms.ToTensor(),
        ])

        self.dataset = torchvision.datasets.ImageFolder(config['data_path'], transform=self.transform)
        self.loaders = config['dataloader_workers'] if 'dataloader_workers' in config else 0
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.loaders)

        # Encoder
        self.encoder = encoders.SoftEncoder(self.cielab, device=self.device)

    def train(self):
        self.network.train()
        for i, batch in enumerate(self.data_loader):
            # Reset the gradients
            self.optimizer.zero_grad()

            # Fetch images
            batch, _ = batch
            batch = batch.type(self.dtype)
            # Fetch input for the network
            x = batch[:, :1, :, :].to(self.device)
            # Forward pass
            predicted = self.network(x)
            # Fetch output and resize
            actual = functional.interpolate(batch[:, 1:, :, :], size=predicted.shape[2:]).to(self.device)
            # Encode the outputs
            labels = self.encoder(actual)

            # Calculate loss
            loss = self.loss(predicted, labels)
            loss.backward()
            print(f"Iter\t{i}\tLoss: {loss}")

            # Optimizer step
            self.optimizer.step()
            # Clean memory
            torch.cuda.empty_cache()



