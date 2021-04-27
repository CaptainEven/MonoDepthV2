# encoding=utf-8
from __future__ import absolute_import, division, print_function

import importlib
from collections import OrderedDict

import torchvision.models as models

from layers import *


def class_for_name(module_name, class_name):
    """
    :param module_name:
    :param class_name:
    :return:
    """
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)

    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        """
        :param num_in_layers:
        :param num_out_layers:
        :param kernel_size:
        :param stride:
        """
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        """
        :param num_in_layers:
        """
        super(get_disp, self).__init__()

        self.conv1 = nn.Conv2d(num_in_layers, 1, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        :param x:
        :return:
        """
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)

        # https://github.com/OniroAI/MonoDepth-PyTorch/issues/13
        return 0.3 * self.sigmoid(x)
        # return self.sigmoid(x)


class EncoderDecoder(nn.Module):
    def __init__(self, num_in_layers, encoder='resnet18', pretrained=True):
        """
        :param num_in_layers:
        :param encoder:
        :param pretrained:
        """
        super(EncoderDecoder, self).__init__()

        assert encoder in ['resnet18', 'resnet34', 'resnet50', \
                           'resnet101', 'resnet152'], \
            "Incorrect encoder type"

        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]

        # Define Resnet backbone: pre-trained or not
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(num_in_layers, 64,
                                       kernel_size=(7, 7), stride=(2, 2),
                                       padding=(3, 3), bias=False)
        else:
            self.firstconv = resnet.conv1  # H/2

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.encoder1 = resnet.layer1  # H/4
        self.encoder2 = resnet.layer2  # H/8
        self.encoder3 = resnet.layer3  # H/16
        self.encoder4 = resnet.layer4  # H/32

        # decoder
        self.upconv6 = upconv(filters[3], 512, 3, 2)
        self.iconv6 = conv(filters[2] + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(filters[1] + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(filters[0] + 128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 1)  #
        self.iconv3 = conv(64 + 64 + 1, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + 1, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + 1, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        # encoder
        x_conv_first = self.firstconv(x)
        x = self.firstbn(x_conv_first)
        x = self.firstrelu(x)
        x_pool_first = self.firstmaxpool(x)

        x1 = self.encoder1(x_pool_first)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # skips
        skip1 = x_conv_first
        skip2 = x_pool_first
        skip3 = x1
        skip4 = x2
        skip5 = x3

        # decoder
        upconv6 = self.upconv6(x4)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=1, mode='bilinear', align_corners=True)
        self.disp4 = nn.functional.interpolate(self.disp4, scale_factor=0.5, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        # return self.disp1, self.disp2, self.disp3, self.disp4

        self.outputs = {}
        self.outputs[("disp", 0)] = self.disp1
        self.outputs[("disp", 1)] = self.disp2
        self.outputs[("disp", 2)] = self.disp3
        self.outputs[("disp", 3)] = self.disp4

        return self.outputs


class StereoNet(nn.Module):
    """
    PyTorch module for a resnet encoder
    """

    def __init__(self, num_layers,
                 pre_trained,
                 num_input_images=1,
                 num_ch_enc=[64, 64, 128, 256, 512],  # encode params
                 scales=range(4),
                 num_output_channels=1,
                 use_skips=True):  # decoder params
        """
        :param num_layers:
        :param pre_trained:
        :param num_input_images:
        """
        super(StereoNet, self).__init__()

        self.num_ch_enc = np.array(num_ch_enc)

        # ---------- encoder
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pre_trained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pre_trained)
        print('Using models from {:s}.'.format('pre-trained' if pre_trained else 'stratch.'))

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        ## ---------- decoder
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        # self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def get_encoder_features(self, input_image):
        """
        :param input_image:
        :return:
        """
        ## ---------- encoder
        self.features = []

        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        self.features.append(self.encoder.relu(x))  # /2
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))  # /4
        self.features.append(self.encoder.layer2(self.features[-1]))  # /8
        self.features.append(self.encoder.layer3(self.features[-1]))  # /16
        self.features.append(self.encoder.layer4(self.features[-1]))  # /32

        return self.features

    def get_decoder_depths(self, input_features):
        """
        :param input_features:
        :return:
        """
        # ---------- decoder
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            ## the last layer of each scale
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

    def forward(self, input_image):
        """
        :param input_image:
        :return:
        """
        ## ---------- encoder
        self.features = []

        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        ## ---------- decoder
        self.outputs = {}

        x = self.features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [self.features[i - 1]]

            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            ## the last layer of each scale
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

        ## ----- Get features
        # self.get_encoder_features(input_image)

        # # ----- Get outputs
        # self.get_decoder_depths(self.features)

        # return self.outputs
