import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset, DataLoader

# Lightning module
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import JaccardIndex
from torchmetrics.classification import Dice

class DoubleConv(nn.Module):
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

class ResNetUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_prob=0.0, pretrained=True):
        super(ResNetUnet, self).__init__()
        self.encoder = torchvision.models.resnet50(pretrained=pretrained)

        self.layer0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.reduce_channels = nn.Conv2d(2048, features[-1], kernel_size=1)

        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.reduce_channels(x5)
        x = self.bottleneck(x)

        skip_connections = [x1, x2, x3, x4][::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

class SegmentLightning(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate, scheduler_name, scheduler_params, train_dataset, val_dataset, batch_size=4, num_workers=2):
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
        img_grid = torchvision.utils.make_grid(data)  # sample data test in TB
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
                'monitor': 'val_loss',
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

def main():
    # Example code for running and testing the model
    image_data = torch.randn((8, 3, 256, 256))  # Example image data with batch_size=8, channels=3, height=256, width=256
    model = ResNetUnet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])
    pred = model(image_data)
    print(pred.shape)

if __name__ == "__main__":
    main()