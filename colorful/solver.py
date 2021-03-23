import colorful.model
import util

import torch
from torch.nn import functional
import torchvision
from torch.utils.data import DataLoader
from colorful import cielab, encoders, CELoss
from colorful.log.loss import LossLogger
from colorful.log.out import OutputLogger
from colorful.log.validator import Validator

import matplotlib.pyplot as plt
from skimage import color
import os

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
        self.progress = 'progress_every' in config
        if self.progress:
            self.progress_every = config['progress_every']
            self.progress_dir = None if 'progress_dir' not in config else config['progress_dir']

        # Network
        self.network = colorful.model.Colorful().type(self.dtype).to(self.device)

        self.start_iteration = 0
        if 'model_file' in config:
            if 'start_iteration' in config:
                self.start_iteration = config['start_iteration']
            self.network = colorful.model.Colorful()
            self.network.load_state_dict(torch.load('tmp/snapshots/23_03(05:06:05)-19000_9896.pth'))
            self.network.eval()
            self.network = self.network.type(self.dtype).to(self.device)
            self.resume = True
        else:
            self.resume = False
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
            util.rgb2lab(),
            torchvision.transforms.ToTensor(),
        ])

        self.dataset = torchvision.datasets.ImageFolder(config['data_path'], transform=self.transform)
        self.loaders = config['dataloader_workers'] if 'dataloader_workers' in config else 0
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.loaders)
        self.val_dataset = torchvision.datasets.ImageFolder(config['val_data_path'], transform=self.transform)

        # Encoder
        self.encoder = encoders.SoftEncoder(self.cielab, device=self.device)


        self.listeners = []
        self.listeners.append(LossLogger(format="({0}, {1})\n", output="./tmp/desmos.log", append=self.resume))
        self.listeners.append(OutputLogger(self.iterations, self.batch_size, output="./tmp/out.log", append=self.resume))
        self.listeners.append(OutputLogger(self.iterations, self.batch_size, append=self.resume))

        # Validator
        self.listeners.append(
            Validator(
                config['validate_every'],
                config['val_data_size'],
                self,
                './tmp/val.log',
                config['snapshot_every'],
                config['snapshot_dir'],
                append=self.resume
            ))

    def train(self):
        self.network.train()
        for i, batch in enumerate(self.data_loader, self.start_iteration):
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
            for logger in self.listeners:
                logger(i, loss.item(), self.network)

            # Optimizer step
            self.optimizer.step()

            if self.progress:
                if i % self.progress_every == 0:
                    with torch.no_grad():
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                        image = batch[0].to(self.device)

                        l = image[:1, :, :]
                        predicted = self.network.forward_colorize(l.view(1, *l.shape))

                        image_normal = color.lab2rgb(image.cpu().permute(1, 2, 0))

                        image_gs = torch.cat(3 * [l.cpu()]).permute(1, 2, 0)
                        image_gs /= 100

                        image_colorized = color.lab2rgb(predicted.cpu())
                        ax1.imshow(image_gs)
                        ax2.imshow(image_normal)
                        ax3.imshow(image_colorized)

                        if self.progress_dir:
                            plt.savefig(os.path.join(self.progress_dir, f"{i}.png"))
                        else:
                            plt.show()
                        plt.close()

            # Clean memory
            torch.cuda.empty_cache()





