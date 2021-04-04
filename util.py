import torch
from torch.utils.data import Sampler
from torchvision.transforms import functional
from skimage import color
import matplotlib.pyplot as plt
import random
import math

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
        scale = self.size/min_len
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

def display_lab(img):
    img = color.lab2rgb(img)
    plt.imshow(img)
    plt.show()
    plt.close()

class ShuffledFilterSampler(Sampler[int]):
    def __init__(self, indexes=None, indexes_file=None):
        self.indexes = None
        if indexes and indexes_file:
            raise RuntimeError("Only one argument has to be defined!")

        if indexes:
            self.indexes = indexes
        if indexes_file:
            self.indexes = []
            with open(indexes_file, 'r') as file:
                for line in file:
                    self.indexes.append(int(line.strip()))
        if self.indexes is None:
            raise RuntimeError("One of two arguments has to be defined!")
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.indexes)

    def __iter__(self):
        self.shuffle()
        return (i for i in self.indexes)

class SubsetFilterSampler(Sampler[int]):
    def __init__(self, samples, indexes=None, indexes_file=None):
        self.indexes = None
        if indexes and indexes_file:
            raise RuntimeError("Only one argument has to be defined!")

        if indexes:
            self.indexes = indexes
        if indexes_file:
            self.indexes = []
            with open(indexes_file, 'r') as file:
                for line in file:
                    self.indexes.append(int(line.strip()))
        if self.indexes is None:
            raise RuntimeError("One of two arguments has to be defined!")

        assert len(self.indexes) > samples

        self.total_samples = len(self.indexes)
        self.samples = samples
        self.increment = int(math.floor(self.total_samples/self.samples))
        self.max_i = self.samples * self.increment
        self.indexes = self.indexes[0:self.max_i:self.increment]

    def __iter__(self):
        return (i for i in self.indexes)


def is_grayscale(img, threshold_val = 10, threshold_percentage = .9):
    def check_channel(channel):
        # Checks if threshold percentage of pixels are in range -threshold < channel < threshold
        total_elements = channel.shape[0] * channel.shape[1]
        threshold_elements = total_elements * threshold_percentage
        lo = -threshold_val <= channel
        hi = channel <= threshold_val
        in_range = torch.logical_and(hi, lo)
        non_zero = torch.count_nonzero(in_range)
        return non_zero > threshold_elements
    return check_channel(img[1]) and check_channel(img[2])