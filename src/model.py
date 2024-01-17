import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet101

class DoubleConv(nn.Module):
    """
    Double convolution block with batch normalization and ReLU activation.
    
    Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, x):
        return self.conv(x)

class BaseUnet(nn.Module):
    """
    Base UNet model for semantic segmentation of kelp.
    
    Parameters:
        - in_channels (int): Number of input channels (default is 3 for Landsat bands).
        - out_channels (int): Number of output channels (default is 1 for binary segmentation).
        - features (list): List of features for each level of the UNet (default is [64, 128, 256, 512]).
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_prob=0.0):
        super(BaseUnet, self).__init__()
        self.in_channels = in_channels
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_prob=dropout_prob))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the UNet model.
        
        Parameters:
            - x (torch.Tensor): Input tensor.
            
        Returns:
            - torch.Tensor: Output tensor.
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)  # Fix dimension along channels
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
def main():
    # Example code for running and testing the models
    image_data = torch.randn((4, 9, 350, 350))  # Example image data
    model = BaseUnet(in_channels=9, out_channels=1)
    pred = model(image_data)
    print(pred.shape)
    
# Check if the script is being run as the main program
if __name__ == "__main__":
    # Call the main function
    main()
