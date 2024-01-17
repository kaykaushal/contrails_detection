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
        img_name = self.dataframe.iloc[idx, 0]  # Assuming the first column contains file names
        img_path = self.root_dir / img_name
        # Load and preprocess the image and mask
        image, mask = self.load_image(img_path)

        # Select a specific image from the numpy sequence (adjust index as needed)
        selected_image = image[..., self.sequence_index]
        # Change shape for tensor operation
        image_bands = selected_image.transpose((1, 2, 0))
        mask = mask.squeeze().astype(np.float32)
        # Tensor
        if self.transform:
            image_bands = self.transform(image_bands)
            mask = self.transform(mask)

        return image_bands, mask

    def load_image(self, img_path):
        # Convert the Path objects to strings
        mask_path = f"{img_path}/human_pixel_masks.npy"
        image = np.array([np.load(f"{img_path}/band_{band:02d}.npy") for band in range(8, 17)])
        image = np.array(image)
        mask = np.load(mask_path)
        return image, mask
    

class ContrailsDatasetV2(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, sequence_index=0):
        self.dataframe = dataframe
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sequence_index = sequence_index

        # Define the transformation for loading images
        self.loader = DatasetFolder(
            root=self.root_dir,
            extensions=('.npy',),
            loader=self.load_image,
            transform=transforms.Compose([transforms.Lambda(lambda x: x[0][..., sequence_index]),
                                          transforms.Lambda(lambda x: x.transpose((1, 2, 0)))])
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # Assuming the first column contains file names
        img_path = self.root_dir / img_name

        # Use the defined loader to load and preprocess the image and mask
        image, mask = self.loader(img_path)

        mask = mask.squeeze().astype(np.float32)
        # Tensor
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def load_image(self, file_path):
        # Load and preprocess the image and mask
        files = [f for f in os.listdir(file_path) if not f.startswith('.')]
        image = []
        mask_path = file_path.replace('band_', 'human_pixel_masks.')
        mask = np.load(mask_path)

        for band in range(8, 17):
            band_path = file_path.replace('band_', f'band_{band:02d}.')
            img = np.load(band_path)
            image.append(img)

        image = np.array(image)
        return image, mask
