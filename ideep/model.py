import torch
from torch import nn
from torchvision.transforms import functional

from util.module import Network
from colorful.model import Colorful


class Colorizer(Network):
    def __init__(self, colorful_path, dim_in=1, dim_out=2, input_min=256, bias=True):
        super(Colorizer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.input_min = input_min

        def create_block(dim_in, dim_out, n, kernel_size=3, stride=1, padding=1, dilation=1, prepend_relu=False, norm=True):
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
            if norm:
                block.append(nn.BatchNorm2d(dim_out))
            return nn.Sequential(*block)

        colorful_model = Colorful()
        state_dict = torch.load(colorful_path)
        colorful_model.load_state_dict(state_dict)
        colorful_model.eval()

        # Convolutional blocks are pretrained
        model1_children = list(colorful_model.model1.children())
        self.conv1 = nn.Sequential(*model1_children[:-3])
        self.conv1_down = nn.Sequential(*model1_children[-3:])
        model2_children = list(colorful_model.model2.children())
        self.conv2 = nn.Sequential(*model2_children[:-3])
        self.conv2_down = nn.Sequential(*model2_children[-3:])
        model3_children = list(colorful_model.model3.children())
        self.conv3 = nn.Sequential(*model3_children[:-3])
        self.conv3_down = nn.Sequential(*model3_children[-3:])
        self.conv4 = colorful_model.model4
        self.conv5 = colorful_model.model5
        self.conv6 = colorful_model.model6
        self.conv7 = colorful_model.model7

        # Fetch conv8 parts
        conv8_children = list(colorful_model.model8.children())
        # Get the upsampling layer
        self.conv8up = conv8_children[0]
        # Ignore the upsampling layer as well as the 256->313 mapping layer
        self.conv8 = nn.Sequential(*conv8_children[1:-1])
        # Create short layer
        self.conv3_8short = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias)

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

        self.global_hints = create_block(316, 512, 4, kernel_size=1, padding=0)

        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias),
            nn.Tanh()
        )

    def forward_colorize(self, input_l, global_hints=None, normalized=False):
        train = self.training
        self.train(False)
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
            out = self.forward_train(l_in, global_hints, normalized=normalized)
            ab = self.unnormalize_ab(out)

            # Get L and ab values
            l = input_l[0]
            ab = ab[0]
            # Resize ab to be the size of the input
            ab = functional.resize(ab, l.shape[1:])
            # Create image and permute
            img = torch.cat((l, ab), dim=0)
            img = img.permute(1, 2, 0)
        self.train(train)
        return img

    def forward_train(self, input_l, global_hints=None, normalized=False):
        if not normalized:
            input_l = self.normalize_l(input_l)

        if global_hints == None:
            # Expand dimensions and pull through global hints part of network
            global_hints = torch.zeros((input_l.shape[0], 316), device=input_l.device)

        c1 = self.conv1(input_l)
        c1_down = self.conv1_down(c1)
        c2 = self.conv2(c1_down)
        c2_down = self.conv2_down(c2)
        c3 = self.conv3(c2_down)
        c3_down = self.conv3_down(c3)
        c4 = self.conv4(c3_down)
        global_hints = self.global_hints(global_hints[:, :, None, None])
        c5_in = c4 + global_hints.expand_as(c4)
        c5 = self.conv5(c5_in)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8_up = self.conv8up(c7)
        c8_short = self.conv3_8short(c3)
        c8_in = c8_up + c8_short
        c8 = self.conv8(c8_in)
        c9_in = self.conv9up(c8) + self.conv2_9short(c2)
        c9 = self.conv9(c9_in)
        c10_in = self.conv10up(c9) + self.conv1_10short(c1)
        c10 = self.conv10(c10_in)
        return self.out(c10)
