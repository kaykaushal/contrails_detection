import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification import Dice

# src code
#import src.helper as helper
import src.utils as utils
from src.dataset import ContrailsDataset
from src.model import BaseUnet

# Custom LR Scheduler
class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, patience, factor, min_lr, max_epochs):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        super(CustomLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.factor ** (self.last_epoch / self.patience) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch >= self.max_epochs:
            return
        if self.last_epoch - self.num_bad_epochs >= self.patience:
            self.last_epoch = self.last_epoch - self.num_bad_epochs + 1
            self.num_bad_epochs = 0
            self.optimizer.param_groups[0]['lr'] = max(self.min_lr, self.optimizer.param_groups[0]['lr'] * self.factor)


class TrainerLightning(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, train_dataset, val_dataset,
                 scheduler_name, scheduler_params, batch_size, num_workers):
        super(TrainerLightning, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        iou = utils.binary_iou_dice_score(outputs, targets.to(torch.uint8), metric="iou")
        dice_coefficient = utils.binary_iou_dice_score(outputs, targets.to(torch.uint8), metric="dice")
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_iou', dice_coefficient, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice_coefficient, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'iou':iou, 'dice': dice_coefficient}

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        iou = utils.binary_iou_dice_score(outputs, targets.to(torch.uint8), metric="iou")
        dice_coefficient = utils.binary_iou_dice_score(outputs, targets.to(torch.uint8), metric="dice")
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice_coefficient, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_iou': iou, 'val_dice': dice_coefficient}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer)
        scheduler = self.get_scheduler(optimizer)
        return [optimizer], [scheduler]

    def get_scheduler(self, optimizer):
        if self.scheduler_name == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params),
                'monitor': 'val_loss',
                'mode': 'min',
                'factor': 2.5,
                'patience': 5,
                'min_lr': 1e-6
            }
            return scheduler
        elif self.scheduler_name == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **self.scheduler_params)
        elif self.scheduler_name == 'CustomLRScheduler':
            return CustomLRScheduler(optimizer, **self.scheduler_params)
        else:
            return None

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False)
