import torch
import torch.nn as nn
from layernorm import LayerNorm

"""
Definition of the ConvNeXt Architecture block.
The ConvNeXt block is a variant of the ResNeXt block, which is a variant of the ResNet block.
The ConvNeXt block is a combination of a depthwise convolution, a layer normalization, two pointwise convolutions, and a GELU activation function.
As we are working with grayscale images, the first layer of the ConvNeXt block is a standard convolution, while the other layers are depthwise convolutions followed by a pointwise convolution
as in the original architecture. It is not neccesary in firts instance as other layers are computed before this one, but the check is added in case of the neccesity of different variations.
Parameters:
    in_channels: int - number of input channels
    out_channels: int - number of output channels
    stride: int - stride of the convolution. Initial value is 1, which means no downsampling.
    layer_scale_init_value: float - initial value of the layer scale parameter
"""

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, stride=1, layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        
        # Depthwise convolution for other layers
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)  # layer norm
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
       
        # Store the input for the identity (residual) connection
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # Permute to apply channels_last LayerNorm
        x = self.norm(x)  # Correct LayerNorm usage
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # Permute back to channels_first before adding residual connection
        x += identity
        return x
