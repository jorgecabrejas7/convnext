import torch
import torch.nn as nn
from convnext_encoder import ConvNeXtEncoder
from unet_decoder import UNetDecoder


class ConvNeXtUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ConvNeXtUNet, self).__init__()
        self.encoder = ConvNeXtEncoder(in_channels=in_channels)
        self.decoder = UNetDecoder([96, 192, 384, 768], num_classes)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output


if __name__ == '__main__':
    from torchviz import make_dot
    from torchsummary import summary
    model = ConvNeXtUNet()
    
  
    batch_size = 1
    channels = 1
    height = 128        
    width = 128

    # Create a random input tensor
    x = torch.randn(batch_size, channels, height, width)

    # Initialize the model
    model = ConvNeXtUNet()

    # Pass the input through the encoder to get features
    features = model.encoder(x)

    # Print shapes of intermediate features
    for i, feature_map in enumerate(features):
        print(f"Feature map {i} shape:", feature_map.shape)

    # Pass features through the decoder to get the final output
    output = model.decoder(features)

    # Print the final output shape
    print("Final output shape:", output.shape)

# torch.onnx.export(model, x, "model.onnx", verbose=True, input_names=['input'], output_names=['output'])