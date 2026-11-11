"""
U-Net model for the knee X-ray bone segmentation project.

The code below is taken from the deep-pipeline project by Aleksei Tiulpin:
https://github.com/lext/deep-pipeline
  - deeppipeline/segmentation/models/_unet.py   (UNet)
  - deeppipeline/segmentation/models/modules.py (Encoder, Decoder)
  - deeppipeline/common/modules.py              (conv_block_nxn, conv_block_3x3)

The original code is split into several files. Here the needed parts are
collected into this one file, so that model.py works on its own.
The code itself is not changed.

MIT License, Copyright (c) 2019 Aleksei Tiulpin. See the LICENSE file.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# From deeppipeline/common/modules.py
# ---------------------------------------------------------------------------
def conv_block_nxn(ks, inp, out, activation='relu', normalization='BN', bias=True):
    """3x3 ConvNet building block with different activations and normalizations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    if normalization == 'BN':
        norm_layer = nn.BatchNorm2d(out)
    elif normalization == 'IN':
        norm_layer = nn.InstanceNorm2d(out)
    elif normalization is None:
        norm_layer = nn.Sequential()
    else:
        raise NotImplementedError

    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            norm_layer,
            nn.ReLU(inplace=True)
        )

    elif activation == 'selu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            nn.SELU(inplace=True)
        )

    elif activation == 'elu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            nn.ELU(1, inplace=True)
        )
    elif activation is None:
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            norm_layer,
        )


def conv_block_3x3(inp, out, activation='relu', normalization='BN', bias=True):
    return conv_block_nxn(3, inp, out, activation, normalization, bias)


# ---------------------------------------------------------------------------
# From deeppipeline/segmentation/models/modules.py
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Encoder block. For encoder-decoder architecture.
    Conv3x3-Conv3x3-Maxpool 2x2. BatchNorm or other form of normalization is used optionally.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, inp_channels, out_channels, depth=2, activation='relu', normalization='BN'):
        super().__init__()
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(conv_block_3x3(inp_channels, out_channels, activation, normalization))
            else:
                tmp.append(conv_block_3x3(out_channels, out_channels, activation, normalization))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x):
        processed = self.layers(x)
        pooled = F.max_pool2d(processed, 2, 2)
        return processed, pooled


class Decoder(nn.Module):
    """Decoder block. For encoder-decoder architecture.
    Bilinear ups->Conv3x3-Conv3x3. BatchNorm or other form of normalization is used optionally.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, inp_channels, out_channels, depth=2, mode='bilinear', activation='relu', normalization='BN'):
        super().__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(conv_block_3x3(inp_channels, out_channels, activation, normalization))
            else:
                tmp.append(conv_block_3x3(out_channels, out_channels, activation, normalization))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x_big, x):
        x_ups = F.interpolate(x, size=x_big.size()[-2:], mode=self.ups_mode, align_corners=True)
        y = torch.cat([x_ups, x_big], 1)
        y = self.layers(y)
        return y


# ---------------------------------------------------------------------------
# From deeppipeline/segmentation/models/_unet.py
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on the depth and width.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    Parameters
    ----------
    bw : int
        Basic width of the network, which is doubled at each layer.
    depth : int
        Number of layers
    center_depth : int
        Depth of the central block in UNet.
    n_inputs : int
        Number of input channels.
    n_classes : int
        Number of input classes
    activation : str
        Activation function. Can be ReLU, SeLU, or ELU. The latter has parameter 1 by default.
    norm_encoder : str
        Normalization to be used in the encoder.
    norm_decoder : str
        Normalization to be used in the decoder.
    norm_center : str
        Normalization to be used in the center of the network.
    """

    def __init__(self, bw=24, depth=6, center_depth=2, n_inputs=1, n_classes=1,
                 activation='relu', norm_encoder='BN', norm_decoder='BN', norm_center='BN'):
        super().__init__()
        # Preparing the modules dict
        modules = OrderedDict()
        modules['down1'] = Encoder(n_inputs, bw, activation=activation, normalization=norm_encoder)
        # Automatically creating the Encoder based on the depth and width

        mul_out = None
        for level in range(2, depth + 1):
            mul_in = 2 ** (level - 2)
            mul_out = 2 ** (level - 1)
            layer = Encoder(bw * mul_in, bw * mul_out, activation=activation, normalization=norm_encoder)
            modules['down' + str(level)] = layer

        if mul_out is None:
            raise ValueError('The depth parameter is wrong. Cannot determine the output size of the encoder')
            # Creating the center
        modules['center'] = nn.Sequential(
            *[conv_block_3x3(bw * mul_out, bw * mul_out, activation, norm_center) for _ in range(center_depth)]
        )
        # Automatically creating the decoder
        for level in reversed(range(2, depth + 1)):
            mul_in = 2 ** (level - 1)
            layer = Decoder(2 * bw * mul_in, bw * mul_in // 2, activation=activation, normalization=norm_decoder)
            modules['up' + str(level)] = layer

        modules['up1'] = Decoder(bw + bw, bw, activation=activation, normalization=norm_decoder)

        modules['mixer'] = nn.Conv2d(bw, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.__dict__['_modules'] = modules

    def forward(self, x):
        encoded_results = {}
        out = x

        for name in self.__dict__['_modules']:
            if name.startswith('down'):
                layer = self.__dict__['_modules'][name]
                convolved, pooled = layer(out)
                encoded_results[name] = convolved
                out = pooled

        out = self.center(out)

        for name in self.__dict__['_modules']:
            if name.startswith('up'):
                layer = self.__dict__['_modules'][name]
                out = layer(encoded_results['down' + name[-1]], out)

        return self.mixer(out)
