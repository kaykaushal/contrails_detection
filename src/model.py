import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Lightning module
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import JaccardIndex
from torchmetrics.classification import  Dice
from torchmetrics import MetricCollection

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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, x):
        return self.conv(x)


class BaseUnet(nn.Module):
    """
    Base UNet model for semantic segmentation.

    Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - features (list): List of features for each level of the UNet.
        - dropout_prob (float): Dropout probability (default is 0.0).
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[32, 64, 128, 256],
        dropout_prob=0.0,
    ):
        super(BaseUnet, self).__init__()
        self.in_channels = in_channels
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(
                DoubleConv(in_channels, feature, dropout_prob=dropout_prob)
            )
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

## Attention U-Net 
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(AttentionUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.attention = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))
            self.attention.append(AttentionGate(feature, feature, feature // 2))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            attention = self.attention[idx//2](x, skip_connections[idx//2])
            x = torch.cat((attention, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_conv(x)


## Segnment Modeling using Lightning
class SegmentLightning(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate, scheduler_name,
                 scheduler_params, train_dataset, val_dataset,
                 batch_size=4, num_workers=2):
        super(SegmentLightning, self).__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.iou = JaccardIndex(num_classes=2)
        self.dice_coefficient = Dice()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        img_grid = torchvision.utils.make_grid(data) # sample data test in TB
        self.logger.experiment.add_image("contrails_images", img_grid[0:3], self.global_step)
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        dice_coefficient = self.dice_coefficient(outputs, targets.to(torch.uint8))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice_coefficient, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'dice': dice_coefficient}

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        iou = self.iou(outputs, targets.to(torch.uint8))
        dice_coefficient = self.dice_coefficient(outputs, targets.to(torch.uint8))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice_coefficient, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'iou': iou, 'dice': dice_coefficient}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = self.get_scheduler(optimizer)
        return [optimizer], [scheduler]

    def get_scheduler(self, optimizer):
        if self.scheduler_name == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params),
                'monitor': 'val_loss',  # Adjust based on your metric
                'mode': 'min',
                'factor': 0.5,
                'patience': 3,
                'min_lr': 1e-6
            }
            return scheduler
        elif self.scheduler_name == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **self.scheduler_params)
        else:
            return None  # Handle other scheduler types as needed

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

############## Model with Backbone ###############
from torchvision import models

class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = DoubleConv(256 + 128, 128)
        self.conv_up2 = DoubleConv(128 + 64, 64)
        self.conv_up1 = DoubleConv(64 + 64, 64)
        self.conv_up0 = DoubleConv(64 + 3, 64)
        
        self.conv_original_size0 = DoubleConv(3, 64)
        self.conv_original_size1 = DoubleConv(64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        x = self.upsample(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_last(x)
        
        return x

## UNet++ Model
class UNetPlus(nn.Module):
    """
    U-Net++ model for semantic segmentation.
    
    Parameters:
        - in_channels (int): Number of input channels (default is 3 for RGB images).
        - out_channels (int): Number of output channels (default is 1 for binary segmentation).
        - features (list): List of features for each level of the U-Net++ (default is [32, 64, 128, 256]).
        - dropout_prob (float): Dropout probability (default is 0.0).
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_prob=0.0):
        super(UNetPlus, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net++
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of U-Net++
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
        Forward pass of the U-Net++ model.
        
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

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
### Timeseries Model Training ####
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import  Dice

# Lightning Module for training
class SegmentLightningTS(LightningModule):
    def __init__(self, model, criterion, learning_rate, scheduler_name, scheduler_params, train_dataset, val_dataset, batch_size):
        super(SegmentLightningTS, self).__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.dice_coefficient = Dice()
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        dice_coefficient = self.dice_coefficient(outputs, masks.to(torch.uint8))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_dice', dice_coefficient, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        dice_coefficient = self.dice_coefficient(outputs, masks.to(torch.uint8))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice_coefficient, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)


########### Call main function #################
def main():
    # Example code for running and testing the model
    image_data = torch.randn((4, 9, 350, 350))  # Example image data
    model = BaseUnet(in_channels=9, out_channels=1, features=[32, 64, 128, 256])
    pred = model(image_data)
    print(pred.shape)


# Check if the script is being run as the main program
if __name__ == "__main__":
    # Call the main function
    main()
