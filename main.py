import torchvision
import matplotlib.pyplot as plt
import util
import numpy as np
from colorful.solver import Solver

import torch

transform = torchvision.transforms.Compose([
    util.ShortResize(256),
    torchvision.transforms.RandomCrop(256),
    torchvision.transforms.ToTensor(),
])

# dataset = torchvision.datasets.ImageFolder("/hdd/imagenet/train", transform=transform)

config = {}
config['device'] = "cuda:0"
config['dtype'] = torch.float32
config['lambda'] = 0.5
config['lr'] = 3e-5
config['weight_decay'] = 1e-3
config['iterations'] = 200000
config['batch_size'] = 40
config['data_path'] = "./imagenet/train"
config['dataloader_workers'] = 4

solver = Solver(config)
solver.train()
