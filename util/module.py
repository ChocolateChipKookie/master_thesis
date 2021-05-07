import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from skimage import color
from abc import ABCMeta, abstractmethod
from util import util, sampler, listener


class ShortResize(torch.nn.Module):
    """
        Resizes the image so the shorter side is exactly _size_ pixels long
        Image can be of type Tensor or PIL image
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.interpolation = functional.InterpolationMode.BILINEAR

    def forward(self, img):
        # Fetch the size of the image
        if isinstance(img, torch.Tensor):
            size = list(img.shape[1:])
        else:
            size = list(img.size)[::-1]
        # Get the length of the shorter side
        min_len = min(size)
        # Calculate scale
        scale = self.size / min_len
        for i in range(len(size)):
            size[i] = int(round(size[i] * scale))
        # Resize
        resized = functional.resize(img, size, self.interpolation)
        return resized


class rgb2lab(torch.nn.Module):
    """
        Converts PIL image from the rgb colour space to CIELAB
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        # Fetch the size of the image
        return color.rgb2lab(img)


class Solver(object):
    __metaclass__ = ABCMeta

    def __init__(self, network, config):
        # Save full config
        self.config = config
        self.solver_config = self.config['solver']['config']
        self.derived_config = self.config['solver']['derived']

        # Basic settings
        self.device = torch.device(self.solver_config['device'])
        self.dtype = util.import_attr(self.solver_config['dtype'])
        self.iterations = self.solver_config['iterations']
        self.batch_size = self.solver_config['batch_size']
        self.iteration = 0
        self.restore = self.config['restore']

        # Network
        self.network = network.type(self.dtype).to(self.device)

        # In case restore is defined, restore the state dict and set start iteration
        if self.restore:
            if "state" not in self.config:
                print("Please load config file with state defined, state has to have at least values: iter, state_dict")
                exit()
            state = self.config['state']
            state_dict = torch.load(state['state_dict'])
            self.network.load_state_dict(state_dict)
            self.network.eval()
            self.iteration = state['iter'] + 1

        # Optimizer
        optimizer_data = self.solver_config['optimizer']
        optimizer_class = util.import_attr(optimizer_data['class'])
        self.optimizer = optimizer_class(self.network.parameters(), **optimizer_data['args'])

        # Transform
        transforms = []
        for transform in self.solver_config['transforms']:
            t = util.factory(transform)
            transforms.append(t)
        # Change data type of the tensor
        transforms.append(torchvision.transforms.ConvertImageDtype(dtype=self.dtype))
        # Create composite transform
        self.transform = torchvision.transforms.Compose(transforms)

        # If loaders are defined, set the value, else default
        if 'dataloader_workers' in self.solver_config:
            self.loaders = self.solver_config['dataloader_workers']
        else:
            self.loaders = 0

        self.train_dataset = torchvision.datasets.ImageFolder(self.solver_config['train_path'], transform=self.transform)
        self.sampler = sampler.ShuffledFilterSampler(indexes_file=self.solver_config['train_mask'])

        self.val_dataset = torchvision.datasets.ImageFolder(self.solver_config['val_path'], transform=self.transform)
        self.val_sampler = sampler.SubsetFilterSampler(
            self.solver_config['val_data_size'],
            indexes_file=self.solver_config['val_mask']
        )

        self.listeners = []

        for data in self.solver_config['listeners']:
            listener_class = util.import_attr(data["class"])
            l = listener_class(solver=self, **data["args"])
            if not isinstance(l, listener.SolverListener):
                raise TypeError("Listeners should inherit from SolverListener!")
            self.listeners.append(l)

    def train(self):
        self.network.train()

        while self.iteration < self.iterations:
            data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers=self.loaders
            )

            for i, batch in enumerate(data_loader, self.iteration):
                self.iteration += 1
                # Reset the gradients
                self.optimizer.zero_grad()
                # Fetch images
                batch, _ = batch

                loss = self.calculate_loss(batch)
                loss.backward()

                for logger in self.listeners:
                    logger(i, loss.item())

                # Optimizer step
                self.optimizer.step()

                # Clean memory
                torch.cuda.empty_cache()

    @abstractmethod
    def calculate_loss(self, batch):
        pass


class Network(torch.nn.Module):
    __metaclass__ = ABCMeta
    def __init__(self):
        super().__init__()

    def forward(self, input_l):
        if self.training:
            return self.forward_train(input_l)
        else:
            return self.forward_colorize(input_l)

    @abstractmethod
    def forward_train(self, input_l):
        pass

    @abstractmethod
    def forward_colorize(self, input_l):
        pass
