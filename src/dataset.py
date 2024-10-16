import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from pathlib import Path
import numpy as np
import pandas as pd


class ContrailsDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, sequence_index=0):
        self.dataframe = dataframe
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sequence_index = sequence_index

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.iloc[idx, 0])
        img_path = self.root_dir / img_name
        # Load and preprocess the image and mask
        image, mask = self.load_image(img_path)

        # Select a specific image from the numpy sequence (adjust index as needed)
        selected_image = image[..., self.sequence_index]
        # Change shape for tensor operation
        image_bands = selected_image.transpose((1, 2, 0))
        mask = mask.squeeze().astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        # Tensor
        if self.transform:
            image_bands = self.transform(image_bands)
            #mask = self.transform(mask)

        return image_bands, mask

    def load_image(self, img_path):
        # Convert the Path objects to strings
        mask_path = f"{img_path}/human_pixel_masks.npy"
        image_paths = [img_path / f"band_{band:02d}.npy" for band in range(8, 17)]
        image = np.array([np.load(path) for path in image_paths])
        mask = np.load(mask_path)
        return image, mask


class ContrailsDatasetSeqs(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.iloc[idx, 0])
        img_path = self.root_dir / img_name

        # Load and preprocess the image and mask
        image, mask = self.load_image(img_path)

        # Change shape for tensor operation (C, T, H, W)
        image_bands = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask.squeeze().astype(np.float32)).unsqueeze(0)

        if self.transform:
            image_bands = self.transform(image_bands)

        return image_bands.permute(0, 3, 1, 2), mask #.unsqueeze(0).expand(-1, 8, -1, -1)

    def load_image(self, img_path):
        # Convert the Path objects to strings
        mask_path = f"{img_path}/human_pixel_masks.npy"
        image_paths = [img_path / f"band_{band:02d}.npy" for band in range(8, 17)]
        image = np.array([np.load(path) for path in image_paths])
        mask = np.load(mask_path)
        return image, mask