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


class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


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
            self.state = self.config['state']
            state_dict = torch.load(self.state['state_dict'])
            self.network.load_state_dict(state_dict)
            self.network.eval()
            self.iteration = self.state['iter'] + 1

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
        self.shuffled_val_sampler = sampler.SubsetFilterSampler(
            self.solver_config['val_data_size'],
            indexes_file=self.solver_config['val_mask'],
            shuffle=True
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

        while self.iteration <= self.iterations:
            data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers=self.loaders
            )

            for i, batch in enumerate(data_loader, self.iteration):
                torch.cuda.empty_cache()
                # Check if exit condition is satisfied
                if self.iteration > self.iterations:
                    break
                self.iteration += 1
                # Reset the gradients
                self.optimizer.zero_grad()
                # Fetch images
                batch, _ = batch
                if batch.shape[0] < 2:
                    continue

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

        self.l_mean = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_colorize(*args, **kwargs)

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_colorize(self, *args, **kwargs):
        pass

    def normalize_l(self, in_l):
        return (in_l - self.l_mean) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_mean

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm

    def normalize(self, in_lab):
        in_lab[:, :1] = self.normalize_l(in_lab[:, :1])
        in_lab[:, 1:] = self.normalize_ab(in_lab[:, 1:])
        return in_lab

    def unnormalize(self, in_lab):
        in_lab[:, :1] = self.unnormalize_l(in_lab[:, :1])
        in_lab[:, 1:] = self.unnormalize_ab(in_lab[:, 1:])
        return in_lab
