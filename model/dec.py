import torch
import torch.nn as nn
from . import utils
import math

class DCEC_AutoEncoder(nn.Module):
    def __init__(self, dim_input, dim_code, num_input_channels,
                 num_feature_maps0, num_feature_maps1, num_feature_maps2,
                 dropout_value=0, batchnorm=False):
        super(DCEC_AutoEncoder, self).__init__()
        self.enc0 = utils.ConvBlock(transposed=False,
                            in_channels=num_input_channels,
                            out_channels=num_feature_maps0,
                            kernel_size=5, padding=2, stride=2, output_padding=0,
                            dropout_value=dropout_value, conduct_batchnorm=batchnorm)
        self.enc1 = utils.ConvBlock(transposed=False,
                            in_channels=num_feature_maps0,
                            out_channels=num_feature_maps1,
                            kernel_size=5, padding=2, stride=2, output_padding=0,
                            dropout_value=dropout_value, conduct_batchnorm=batchnorm)
        self.enc2 = utils.ConvBlock(transposed=False,
                            in_channels=num_feature_maps1,
                            out_channels=num_feature_maps2,
                            kernel_size=3, padding=0, stride=2, output_padding=0,
                            dropout_value=dropout_value, conduct_batchnorm=batchnorm)

        side_feature_maps = math.floor((((dim_input + 2 * 2 - (5 - 1) - 1)) / 2) + 1)
        side_feature_maps = math.floor((((side_feature_maps + 2 * 2 - (5 - 1) - 1) / 2)) + 1)
        side_feature_maps = math.floor((((side_feature_maps + 2 * 0 - (3 - 1) - 1) / 2)) + 1)
        self.num_feature_maps = num_feature_maps2
        self.side_feature_maps = side_feature_maps
        self.dim_around_code = self.num_feature_maps * side_feature_maps**2
        # the structure of feature maps just before and just before coding fully-connected layers.

        self.fc1 = nn.Linear(self.dim_around_code, dim_code)
        self.fc2 = nn.Linear(dim_code, self.dim_around_code)

        self.dec2 = utils.ConvBlock(transposed=True,
                            in_channels=num_feature_maps2,
                            out_channels=num_feature_maps1,
                            kernel_size=3, padding=0, stride=2, output_padding=0,
                            dropout_value=dropout_value, conduct_batchnorm=batchnorm)
        self.dec1 = utils.ConvBlock(transposed=True,
                            in_channels=num_feature_maps1,
                            out_channels=num_feature_maps0,
                            kernel_size=5, padding=2, stride=2, output_padding=1,
                            dropout_value=dropout_value, conduct_batchnorm=batchnorm)
        self.dec0 = utils.ConvBlock(transposed=True,
                            in_channels=num_feature_maps0,
                            out_channels=num_input_channels,
                            kernel_size=5, padding=2, stride=2, output_padding=1,
                            dropout_value=dropout_value, conduct_batchnorm=batchnorm)


    def forward(self, x):
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.fc1(x.view(-1, self.dim_around_code))
        code = x
        x = self.fc2(x)
        x = self.dec2(x.view(-1, self.num_feature_maps, self.side_feature_maps, self.side_feature_maps))
        x = self.dec1(x)
        x = self.dec0(x)

        return x, code