import warnings
warnings.filterwarnings('ignore')
import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import config
from src.dataset import ContrailsDataset
from src.model import BaseUnet, SegmentLightning

# Dataset
df = pd.read_csv(config.CSV_FILE, index_col=0)
train_df, valid_df = train_test_split(df[df.Class.isin([0,1])], test_size = .2, random_state=42)

torch.manual_seed(111)
transform = transforms.Compose(
    [
        # Add your desired transformations here
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0)
    ]
)

train_dataset = ContrailsDataset(
    dataframe=train_df,
    root_dir=config.DATA_DIR + "validation",
    transform=transform,
    sequence_index=config.TS,  # nth:4 sequence image
)

valid_dataset = ContrailsDataset(
    dataframe=valid_df,
    root_dir=config.DATA_DIR + "validation",
    transform=transform,
    sequence_index=config.TS,
)


def main():
    model_tr = BaseUnet(in_channels=9, out_channels=1, features=[32, 64, 128, 256])
    criterion = torch.nn.BCEWithLogitsLoss()
    _expr_name = config.NAME + "_Bands_" + config.BANDS
    #Tensorboard
    logger = TensorBoardLogger(config.TB_PATH, name=_expr_name)
    #writer = SummaryWriter(log_dir=os.path.join(config.TB_PATH, _expr_name))
    
    # Scheduler
    scheduler_name = "ReduceLROnPlateau"
    scheduler_params = {"factor": 0.5, "patience": 3, "min_lr": 1e-6}
    unet_baseline_lgt = SegmentLightning(
        model_tr.to(config.DEVICE),
        criterion,
        config.LR_RATE,
        scheduler_name,
        scheduler_params,
        train_dataset,
        valid_dataset,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.MODEL_DIR + _expr_name, monitor="val_loss", save_last=True, save_top_k=3
    )
    trainer_base = pl.Trainer(
        logger=logger,
        precision=config.PRECISION,
        accelerator=config.ACCELERATOR,
        gpus=[0],  # Adjust GPU IDs
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    # Train the model
    trainer_base.fit(unet_baseline_lgt)

if __name__ == "__main__":
    main()