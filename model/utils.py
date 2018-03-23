import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, transposed, in_channels, out_channels, kernel_size, stride, padding, output_padding=0,
                 dropout_value=0, conduct_batchnorm=False):
        super(ConvBlock, self).__init__()
        if transposed:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        batchnorm = nn.BatchNorm2d(out_channels)
        activation = nn.ReLU()
        dropout = nn.Dropout(dropout_value)
        if conduct_batchnorm:
            ls = [conv, batchnorm, activation, dropout]
        else:
            ls = [conv, activation, dropout]
        if dropout_value:
            pass
        else:
            ls.pop(-1)

        self.block = nn.ModuleList(ls)

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x