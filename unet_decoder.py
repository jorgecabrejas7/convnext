import torch
import torch.nn as nn


"""
Definition of the UNet decoder architecture.
The UNet decoder is a variant of the UNet architecture, which is a variant of the encoder-decoder architecture.
The UNet decoder is a combination of multiple upsampling convolutional layers and multiple convolutional layers.
Parameters:
    in_channels_list: list of ints - number of input channels for each layer
    out_channels: int - number of output channels
"""
class UNetDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(len(in_channels_list) - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(in_channels_list[i], in_channels_list[i-1], kernel_size=2, stride=2))
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels_list[i], in_channels_list[i-1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels_list[i-1], in_channels_list[i-1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ))

        self.final_upconv = nn.ConvTranspose2d(in_channels_list[0], in_channels_list[0] // 2, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(in_channels_list[0] // 2, out_channels, kernel_size=1)
        # Additional upconv to ensure final output size is as expected
        self.extra_upconv = nn.ConvTranspose2d(in_channels_list[0] // 2, out_channels, kernel_size=2, stride=2)

    def forward(self, features):
        a = lambda x: [print(a.shape) for a in x]
        a(features)
        x = features[-1]
        for i in range(len(features) - 2, -1, -1):
            x = self.upconvs[len(features) - 2 - i](x)
            x = torch.cat([x, features[i]], dim=1)
            x = self.convs[len(features) - 2 - i](x)
        x = self.final_upconv(x)  # Additional upsampling to match input size
        x = self.final_conv(x)
        x = self.extra_upconv(x)  # Ensure final output size is doubled
        return x