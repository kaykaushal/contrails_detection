import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_image(img_path):
    # Convert the Path objects to strings
    mask_path = f"{img_path}/human_pixel_masks.npy"
    image = []
    # Load and preprocess the image and mask
    bands = range(8, 17)  # Range from band_08 to band_16
    for band in bands:
        band_path = f"{img_path}/band_{band:02d}.npy"
        img = np.load(band_path)
        image.append(img)
    image = np.array(image)
    #fc_image = get_rgb(image[...,4]) # get rgb of nth sequence image
    mask = np.load(mask_path)
    return image, mask

# Custom loss functions (IoU and Dice coefficient)
def calculate_iou(outputs, targets):
    """
    Calculate the Intersection over Union (IoU) for binary segmentation.
    
    Parameters:
        - outputs (torch.Tensor): Model predictions.
        - targets (torch.Tensor): Ground truth labels.
        
    Returns:
        - torch.Tensor: Mean IoU.
    """
    intersection = torch.logical_and(outputs, targets).sum()
    union = torch.logical_or(outputs, targets).sum()
    iou = (intersection + 1e-10) / (union + 1e-10)
    return iou.mean()

def calculate_dice_coefficient(outputs, targets):
    """
    Calculate the Dice coefficient for binary segmentation.
    
    Parameters:
        - outputs (torch.Tensor): Model predictions.
        - targets (torch.Tensor): Ground truth labels.
        
    Returns:
        - torch.Tensor: Mean Dice coefficient.
    """
    intersection = torch.logical_and(outputs, targets).sum()
    dice_coefficient = (2 * intersection + 1e-10) / (outputs.sum() + targets.sum() + 1e-10)
    return dice_coefficient.mean()

def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def get_rgb(img):
    # Constants for normalization
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    # Extract bands from the input image
    band15 = img[15-8, :, :]
    band14 = img[14-8, :, :]
    band11 = img[11-8, :, :]

    # Normalize the bands
    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)

    # Stack the normalized bands to create an RGB image
    false_color = torch.stack([r, g, b], dim=0)
    
    # Clip values to the range [0, 1]
    false_color = torch.clamp(false_color, 0, 1)
    return false_color

# Plot band and mask together
def plot_rgb_and_mask(image_batch, mask_batch, pred_batch=None):
    """
    Function to plot SWIR band image, Ground truth kelp mask, and
    Kelp mask on SWIR band image for all images in the batch.

    Parameters:
        - image_batch (torch.Tensor): Batch of image tensors.
        - mask_batch (torch.Tensor): Batch of mask tensors.
    """
    batch_size = image_batch.shape[0]
    num_cols = 3 if pred_batch is None else 4

    # Create subplots
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(10, 4*batch_size))

    for i in range(batch_size):
        # Extract the ith image and mask from the batch
        rgb_image = get_rgb(image_batch[i]).cpu().numpy().transpose((1, 2, 0))
        mask_band = mask_batch[i, 0, :, :].cpu().numpy()

        # Plot SWIR band image
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f'Image {i + 1}')

        # Plot Ground truth kelp mask
        axes[i, 1].imshow(mask_band, interpolation='none')
        axes[i, 1].set_title(f'Mask {i + 1}')

        # Plot Kelp mask on SWIR band image
        overlay = np.zeros_like(rgb_image)
        overlay[mask_band > 0.5] = 1.0
        axes[i, 2].imshow(rgb_image)
        axes[i, 2].imshow(overlay, cmap='Reds', alpha=.4, interpolation='none')
        axes[i, 2].set_title(f'Combined {i + 1}')

        # Plot Prediction if available
        if pred_batch is not None:
            pred_band = pred_batch[i, 0, :, :].cpu().numpy()
            axes[i, 3].imshow(pred_band, interpolation='none')
            axes[i, 3].set_title(f'Prediction {i + 1}')