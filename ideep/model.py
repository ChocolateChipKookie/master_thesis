import torch
from torch import nn
from torchvision.transforms import functional

from util.module import Network


class Colorizer(Network):
    def __init__(self, dim_in=1, dim_out=2, input_min=256, bias=True):
        super(Colorizer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.input_min = input_min

        def create_block(dim_in, dim_out, n, kernel_size=3, stride=1, padding=1, dilation=1, prepend_relu=False):
            dilation = (dilation, dilation)
            kernel_size = (kernel_size, kernel_size)
            stride = (stride, stride)
            block = []

            if prepend_relu:
                block.append(nn.ReLU(True))

            block += [
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,  bias=bias),
                nn.ReLU(True)
            ]
            for i in range(1, n):
                block += [
                    nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
                    nn.ReLU(True)
                ]
            block.append(nn.BatchNorm2d(dim_out))
            return nn.Sequential(*block)

        # Create convolutional blocks
        self.conv1 = create_block(self.dim_in, 64, 2)
        self.conv2 = create_block(64, 128, 2)
        self.conv3 = create_block(128, 256, 3)
        self.conv4 = create_block(256, 512, 3)
        self.conv5 = create_block(512, 512, 3, dilation=2, padding=2)
        self.conv6 = create_block(512, 512, 3, dilation=2, padding=2)
        self.conv7 = create_block(512, 512, 3)

        # Create conv8 and skip connection
        self.conv8up = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias)
        self.conv3_8short = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv8 = create_block(256, 256, 2, prepend_relu=True)

        # Create conv9 and skip connection
        self.conv9up = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias)
        self.conv2_9short = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        self.conv9 = create_block(128, 128, 1, prepend_relu=True)

        # Create conv10 and skip connection
        self.conv10up = nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias)
        self.conv1_10short = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)
        # Special case because of leaky relu
        self.conv10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
            nn.LeakyReLU(0.2)
        )

        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias),
            nn.Tanh()
        )
        self.sampling = nn.MaxPool2d(2, 2)

    def forward_colorize(self, input_l, normalized=False):
        with torch.no_grad():
            # Get size of original image
            size = input_l.shape[2:]
            # Fetch shorter side and calculate scaling factor
            scaling_factor = self.input_min / min(size)
            # Calculate new size and cast to ints
            in_size = torch.tensor(size) * scaling_factor
            conv_scaling = 8
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
            input_l = self.normalize_l(input_l)

        c1 = self.conv1(input_l)
        c2 = self.conv2(self.sampling(c1))
        c3 = self.conv3(self.sampling(c2))
        c4 = self.conv4(self.sampling(c3))
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8_in = self.conv8up(c7) + self.conv3_8short(c3)
        c8 = self.conv8(c8_in)
        c9_in = self.conv9up(c8) + self.conv2_9short(c2)
        c9 = self.conv9(c9_in)
        c10_in = self.conv10up(c9) + self.conv1_10short(c1)
        c10 = self.conv10(c10_in)
        return self.out(c10)