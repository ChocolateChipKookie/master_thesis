import torch
from torch import nn
from torchvision.transforms import functional

import util.module
from util.module import Network


class Generator(Network):
    def __init__(self, encoder_layout, decoder_layout, input_min = 256):
        super().__init__()
        # Layouts are lists of tuples that describe the encoder and decoder parts of the network
        # Each pair represents the number of output channels and
        self.e_layout = encoder_layout
        self.d_layout = decoder_layout
        self.input_dims = 1
        self.input_min = input_min
        self.out_dims = 2
        self.kernel = (4, 4)

        encoding_layers = []
        decoding_layers = []
        in_layer = self.e_layout[0]
        encoding_layers.append(
            nn.Sequential(
                nn.ZeroPad2d((2, 1, 2, 1)),
                nn.Conv2d(self.input_dims, in_layer[0], kernel_size=self.kernel, stride=in_layer[1]),
                nn.LeakyReLU(0.2)
            )
        )

        previous = in_layer
        for current in self.e_layout[1:]:
            encoding_layers.append(
                nn.Sequential(
                    nn.ZeroPad2d((2, 1, 2, 1)),
                    nn.Conv2d(previous[0], current[0], kernel_size=self.kernel, stride=current[1]),
                    nn.BatchNorm2d(current[0]),
                    nn.LeakyReLU(0.2)
                )
            )
            previous = current

        for i, current in enumerate(self.d_layout, 1):
            counter = self.e_layout[-i]
            decoding_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(previous[0] + counter[0], current[0], kernel_size=self.kernel, stride=current[1], padding=(1, 1)),
                    nn.BatchNorm2d(current[0]),
                    nn.ReLU()
                )
            )
            previous = current

        self.encoding = util.module.ListModule(*encoding_layers)
        self.decoding = util.module.ListModule(*decoding_layers)
        self.out = nn.Conv2d(previous[0], self.out_dims, kernel_size=(1, 1), stride=(1, 1))

    def forward_colorize(self, input_l, normalized=False):
        with torch.no_grad():
            # Get size of original image
            size = input_l.shape[2:]
            # Fetch shorter side and calculate scaling factor
            scaling_factor = self.input_min / min(size)
            # Calculate new size and cast to ints
            in_size = torch.tensor(size) * scaling_factor
            conv_scaling = 64
            in_size = in_size / conv_scaling
            in_size = torch.round(in_size).type(torch.int32)
            in_size *= conv_scaling
            # Resize input image
            l_in = functional.resize(input_l, list(in_size))

            out = self.forward_train(l_in, normalized)
            ab = self.unnormalize_ab(out)

            # Get L and ab values
            l = input_l[0]
            ab = ab[0]
            # Resize ab to be the size of the input
            ab = functional.resize(ab, l.shape[1:])
            # Create image and permute
            img = torch.cat((l, ab), dim=0)
            img = img.permute(1, 2, 0)
            return img

    def forward_train(self, input_l, normalized=False):
        if not normalized:
            x = self.normalize_l(input_l)
        else:
            x = input_l

        connect = []
        for layer in self.encoding:
            x = layer(x)
            connect.append(x)

        for i, layer in enumerate(self.decoding, 1):
            x = torch.cat((connect[-i], x), dim=1)
            x = layer(x)

        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self, layout):
        super().__init__()
        in_channels = 3
        self.kernel = (4, 4)

        self.layers = []
        prev_channels = in_channels
        for i, c in enumerate(layout):
            channels, stride = c
            batch_norm = False if i == 0 else True
            self.layers.append(nn.Conv2d(prev_channels, channels, kernel_size=self.kernel, stride=stride))
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(channels))
            self.layers.append(nn.LeakyReLU(0.2, True))
            prev_channels = channels

        self.layers.append(nn.Conv2d(prev_channels, 1, kernel_size=(1, 1), stride=(1, 1)))
        self.network = nn.Sequential(*self.layers)

    def forward(self, in_img):
        return self.network(in_img)



