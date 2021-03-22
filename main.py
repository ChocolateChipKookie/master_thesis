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
config['lr'] = 1e-5
config['weight_decay'] = 1e-3
config['iterations'] = 200000
config['batch_size'] = 40

config['data_path'] = "./imagenet/tmp"
config['dataloader_workers'] = 4

config['validate_every'] = 1000
config['val_data_path'] = "./imagenet/val"
config['val_data_size'] = 1000
config['snapshot_every'] = 5
config['snapshot_dir'] = "./tmp/snapshots"


solver = Solver(config)
solver.train()
