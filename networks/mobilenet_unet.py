# encoding=utf-8

from collections import OrderedDict

import torch
import torch.nn as nn
from mobile import MobileNet
from torchsummary import summary


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels:
        :param out_channels:
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        :param in_channels:
        :param out_channels:
        :param mid_channels:
        """
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return self.double_conv(x)


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    """
    :param filter_in:
    :param filter_out:
    :param kernel_size:
    :param groups:
    :param stride:
    :return:
    """
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in,
                           filter_out,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=pad,
                           groups=groups,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


class mobilenet(nn.Module):
    def __init__(self, in_channels):
        """
        :param in_channels:
        """
        super(mobilenet, self).__init__()
        self.model = MobileNet(in_channels)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        out3 = self.model.layer1(x)
        out4 = self.model.layer2(out3)
        out5 = self.model.layer3(out4)
        # out6 = self.model.layer4(out5)

        return out3, out4, out5  # , out6


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels:
        :param out_channels:
        """
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        print('In channels: {:d}.'.format(self.in_channels))
        print('Out channels: {:d}.'.format(self.out_channels))

        # ---------------------------------------------------#
        #   64,64,256；32,32,512；16,16,1024
        # ---------------------------------------------------#
        self.backbone = mobilenet(in_channels)  # Using mobilenet as backbone

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = DoubleConv(1024, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = DoubleConv(512, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        # nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = DoubleConv(128, 64)

        # Define output layers
        self.out0 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out1 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.out2 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.out3 = nn.Conv2d(1024, out_channels, kernel_size=1)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        out = {}

        # Encoder backbone: get features of different scales by the encoder(backbone)
        # x0: 1/16
        # x1: 1/8
        # x2: 1/4
        x2, x1, x0 = self.backbone.forward(x)

        P5 = self.up1(x0)  # 1/8
        P5 = self.conv1(P5)  # P5: 26x26x512

        P4 = x1
        P4 = torch.cat([P4, P5], axis=1)  # P4: 1/8
        out['3'] = self.out3(P4)

        P4 = self.up2(P4)  # 1/4
        P4 = self.conv2(P4)
        P3 = x2  # 1/4
        P3 = torch.cat([P4, P3], axis=1)
        out['2'] = self.out2(P3)

        P3 = self.up3(P3)  # 1/2
        P3 = self.conv3(P3)
        out['1'] = self.out1(P3)

        P3 = self.up4(P3)  # 1/1
        P3 = self.conv4(P3)

        out['0'] = self.out0(P3)

        return out


if __name__ == '__main__':
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=3, out_channels=1)
    model.to(device)
    summary(model, input_size=(3, 320, 1024))  # 768, 448

    # Forward for an input tensor
    input = torch.ones([1, 3, 320, 1024]).to(device)
    out = model.forward(input)

    out = sorted(out.items(), key=lambda x:x[0])  # Ascending order of
    for i, (layer_name, layer) in enumerate(out):
        if i == 0:
            print(layer.shape, 'Scale 0: 1/1')
        elif i == 1:
            print(layer.shape, ' Scale 1: 1/2')
        elif i == 2:
            print(layer.shape, '  Scale 2: 1/4')
        elif i == 3:
            print(layer.shape, '  Scale 3: 1/8')
