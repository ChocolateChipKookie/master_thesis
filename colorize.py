from util import util

import torch
from torchvision.transforms import PILToTensor, functional

import os
import json
from glob import glob
import argparse

from PIL import Image
from skimage import color
import numpy as np

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Colorize image")
    parser.add_argument('config', metavar='config', type=str, help="Config file path")
    parser.add_argument('--out', metavar='out', type=str,  help="Output file, by default the colorized image will be put together with the original file")
    parser.add_argument('--prefix', metavar='prefix', type=str, default="color_",  help="Colorized file prefix")
    parser.add_argument('file_names', metavar='files', type=str, nargs="+", help="Files to colorize")

    # Get args
    args = parser.parse_args()
    # Fetch file paths
    files = []
    for arg in args.file_names:
        files += glob(arg)
    # Fetch config
    with open(args.config, "r") as file:
        config = json.load(file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Create model and load parameters
    model = util.factory(config["colorizer"])
    state_dict = torch.load(config['state_dict'])
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    # Create transformer
    transform = PILToTensor()

    # Don't calculate gradients
    with torch.no_grad():
        # For file in files
        for file in files:
            # Get file name and path
            f = os.path.split(file)
            name = args.prefix + f[1]
            # Create colorized path
            dir = args.out if args.out else f[0]
            save_path = os.path.join(dir, name)
            # Get image
            with Image.open(file) as img:
                mode = img.mode
                img = transform(img)

            channels, width, height = img.shape
            if mode in ["RGBA", "RGB"]:
                # If the mode is rgb or rgba, convert to lab and fetch l channel
                if mode == "RGBA":
                    # If rgba, remove alpha channel
                    img = img[:3]
                rgb = img.permute(1, 2, 0)
                lab = torch.tensor(color.rgb2lab(rgb))
                l_in = lab[:, :, 0]
            elif mode == "L":
                # Image is grayscale
                img = img / 255 * 100
                l_in = img[0]
            else:
                raise NotImplementedError("The colorization for this data mode is not implemented!")

            # Currently l_in is a tensor of shape width x height
            # Reshape to network size

            l_in = l_in.reshape((1, 1, width, height))
            l_in = l_in.type(torch.float)
            predicted = model.forward_colorize(l_in.to(device))
            rgb = color.lab2rgb(predicted.cpu())
            img = Image.fromarray(np.uint8(rgb*255))
            img.save(save_path)
