import torch
from torch import nn

import util.module
from util.module import Network


class Generator(Network):
    def __init__(self, encoder_layout, decoder_layout):
        super().__init__()
        # Layouts are lists of tuples that describe the encoder and decoder parts of the network
        # Each pair represents the number of output channels and
        self.e_layout = encoder_layout
        self.d_layout = decoder_layout
        self.input_dims = 1
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
            out = self.forward_train(input_l, normalized)
            ab = self.unnormalize_ab(out)
            img = torch.cat((input_l, ab), dim=1)
            img = img.permute(0, 2, 3, 1)
            return img[0]

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



