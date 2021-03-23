import torch
from torchvision.transforms import functional
import numpy as np
from skimage import color
import warnings
import matplotlib.pyplot as plt

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

class Lab2rgb(torch.nn.Module):
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
