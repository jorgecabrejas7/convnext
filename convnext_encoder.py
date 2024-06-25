import torch
import torch.nn as nn
from convnext_block import ConvNeXtBlock
from layernorm import LayerNorm



"""
Definition of the full encoder architecture for the ConvNext model.
The ConvNext encoder is a variant of the UNet encoder, which is a variant of the ResNet encoder.
The ConvNext encoder is a combination of a stem, 3 intermediate downsampling convolutional layers, and 4 feature resolution stages, each consisting of multiple ConvNeXt blocks.
Parameters:
    in_channels: int - number of input channels
    depths: list of 4 ints - number of ConvNeXt blocks in each stage
    dims: list of 4 ints - number of channels in each stage

"""
class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_channels=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super(ConvNeXtEncoder, self).__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple ConvNeXt blocks
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i], dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

    def forward(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features