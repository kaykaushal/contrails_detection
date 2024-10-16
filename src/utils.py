import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
import seaborn as sns
import xarray as xr
from pathlib import Path
import dask.array as da
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score


################## Data Collection ##############
# npy image loading using numpy
def load_image(img_path):
    mask_path = f"{img_path}/human_pixel_masks.npy"
    image = []
    bands = range(8, 17)  # Range from band_08 to band_16
    for band in bands:
        band_path = f"{img_path}/band_{band:02d}.npy"
        img = np.load(band_path)
        image.append(img)
    image = np.array(image)
    mask = np.load(mask_path)
    return image, mask

# Load image data and prepare into xarray
def images_list(img_dir, df):
    image_list = []
    mask_list = []

    for index, row in df.iterrows():
        img_name = row['Image_ID']

        img_path = Path(img_dir) / str(img_name)

        # Convert the Path objects to strings
        mask_path = f"{img_path}/human_pixel_masks.npy"
        image_paths = [img_path / f"band_{band:02d}.npy" for band in range(8, 17)]

        # Load image and mask
        image = np.array([np.load(str(path)) for path in image_paths])
        mask = np.load(mask_path)

        image_list.append(image[..., 4])  # Select only the 5th image from the sequence
        mask_list.append(mask)

    return image_list, mask_list

# Image to Xarray dataset
def load_images_to_dataset(img_dir, df):
    img_list, mask_list = images_list(img_dir, df)
    
    # Stack images along the band dimension
    img_concatenated = np.stack(img_list, axis=-1)
    print(img_concatenated.shape)
    # Stack masks along the band dimension
    img_ids = np.repeat(df['Image_ID'], img_concatenated.shape[-1] // len(df))

    # Stack masks along the band dimension
    mask_concatenated = np.stack(mask_list, axis=-1)  # Stack along the first axis
    mask_concatenated = np.squeeze(mask_concatenated, axis=2)
    print(mask_concatenated.shape)

    # Create xarray dataset
    data_vars = {}
    for i in range(img_concatenated.shape[0]):
        data_vars[f'band_{i+8}'] = (('y', 'x', 'band'), img_concatenated[i])
    data_vars['mask'] = (('y', 'x', 'band'), mask_concatenated)

    # Add image IDs as a coordinate
    coords = {'y': np.arange(img_concatenated.shape[1]),
              'x': np.arange(img_concatenated.shape[2]),
              'image_id': img_ids}

    dataset = xr.Dataset(data_vars, coords=coords)

    return dataset

####################### Data Prepration ####################

def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])

#ref: Ash Color Scheme (page 7): https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf
def get_rgb(img):
    _T11_BOUNDS = (0.9, 1.2)
    _CLOUD_TOP_TDIFF_BOUNDS = (-0.015, 0.02)
    _TDIFF_BOUNDS = (-0.015, 0.008)

    band15 = img[15 - 8, :, :]
    band14 = img[14 - 8, :, :]
    band11 = img[11 - 8, :, :]
    band13 = img[13 - 8, :, :]

    r = normalize_range(band15 - band13, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band11, _T11_BOUNDS)

    false_color = torch.stack([r, g, b], dim=0)

    false_color = torch.clamp(false_color, 0, 1)
    return false_color

def get_gaussain_rgb(img):
    r = img[0, :, :]
    g = img[2, :, :]
    b = img[4, :, :]

    false_color = torch.stack([r, g, b], dim=0)
    false_color = torch.clamp(false_color, 0, 1)
    return false_color

# water vapour bands
def get_wv_bands(img):
    img_wv = img[0:3, :, :]
    return img_wv

def get_wv_ash(img):
    wv_band =  img[2, :, :]- img[0, :, :]
    ash_band = img[7, :, :]- img[6, :, :]
    cloud_band = img[3, :, :]
    
    tf_img = torch.stack([wv_band, ash_band, cloud_band], dim=0)
    tf_img = torch.clamp(tf_img, 0, 1)
    return tf_img

def top_feature_bands(img):
    tf_img = torch.stack([img[0, :, :], img[3, :, :], img[7, :, :], img[8, :, :]], dim=0)
    tf_img = torch.clamp(tf_img, 0, 1)
    return tf_img

def get_contrails_bands(img):
    tf_img = torch.stack([img[0, :, :], img[3, :, :], img[5, :, :], img[6, :, :], img[7, :, :]], dim=0)
    return tf_img


def get_ashrgb_tfb(img):
    rgb = get_rgb(img)
    _T11_BOUNDS = (0.9, 1.2)
    # Resize rgb to match the shape of other tensors
    f1 = img[0, :, :]
    f2 = img[3, :, :]
    f3 = img[5, :, :]
    imfb = torch.stack([f1, f2, f3], dim=0)
    out_img = torch.cat([rgb, imfb], dim=0)
    st_img = torch.clamp(out_img, 0, 1)
    return st_img
################ Gaussian Preprocessing #############################
from torchvision import transforms

def gaussian_filter_normalization(image, kernel_size=5, sigma=0.4, k=0.1):
    """Enhances contrast using Gaussian filter normalization (PyTorch).
    
    Args:
        image (torch.Tensor): Input tensor of shape (bands, height, width)
        kernel_size (int): Size of the Gaussian kernel (default=5).
        sigma (float): Standard deviation of the Gaussian kernel (default=1.0).
        k (float): Constant added to local standard deviation to prevent division by zero (default=0.1).

    Returns:
        torch.Tensor: Normalized tensor of the same shape as input.
    """

    # Make sure image is a torch Tensor
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)

    num_bands = image.shape[0]  # Get the number of channels (bands) from the first dimension
    for band_idx in range(num_bands):
        band = image[band_idx, :, :].unsqueeze(0).unsqueeze(0)  # Extract band and add batch and channel dimensions
        smoothed = transforms.functional.gaussian_blur(band, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        local_std = torch.sqrt(transforms.functional.gaussian_blur((band - smoothed) ** 2, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma)))
        normalized_band = (band - smoothed) / (local_std + k)
        normalized_band = torch.clamp(normalized_band, -2, 2)
        image[band_idx, :, :] = normalized_band.squeeze(0).squeeze(0)  # Remove the extra dimensions and put it back in the original image

    return image

def contrail_detection_preprocessing(image, kernel_size=7, sigma=1.0, k=0.1):
    """
    Preprocess a 9-band image for contrail detection using Gaussian filter normalization.
    
    Args:
        image (torch.Tensor): Input tensor of shape (9, height, width).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.
        k (float): Small constant to prevent division by zero.
    
    Returns:
        torch.Tensor: Preprocessed tensor of shape (3, height, width).
    """
    
    # Ensure image has 9 bands
    if image.shape[0] != 9:
        raise ValueError("Input image must have 9 bands.")
    
    # Calculate TD and TDi for each channel
    TD_R = image[6] - image[4]
    TD_Ri = -image[6]
    TDR = torch.stack((TD_R, TD_Ri), dim=0)

    TD_G = image[5] - image[2]
    TD_Gi = -image[2]
    TDG = torch.stack((TD_G, TD_Gi), dim=0)

    TD_B = image[2] - image[0]
    TD_Bi = -image[0]
    TDB = torch.stack((TD_B, TD_Bi), dim=0)

    # Concatenate to form the final image
    final_image = torch.cat((TDR, TDG, TDB), dim=0)

    # Normalize the final image using Gaussian filter normalization
    normalized_image = gaussian_filter_normalization(final_image, kernel_size=kernel_size, sigma=sigma, k=k)

    # Select only the first 3 bands (since each TD channel was expanded by 2 bands)
    return normalized_image

def get_gauss_cont_rgb(img):
    ash_rgb = get_rgb(img)
    gauss_contr_bands = contrail_detection_preprocessing(img)
    final_bands = torch.cat([ash_rgb, gauss_contr_bands], dim=0)
    final_image = torch.clamp(final_bands, 0, 1)
    return final_image

## Combined ASH RGB and Gaussian TD/TDi bands
def compute_TDi_ashrgb(img, T_threshold):
    """
    Compute the Temperature Difference (TD) and Temperature Difference Index (TDi) from an image tensor.

    Parameters:
    - img: Image tensor containing multiple bands (C x H x W)
    - T_threshold: Threshold temperature for TDi calculation, float

    Returns:
    - img_with_TD_TDi: Image tensor with additional TD and TDi channels (C+2 x H x W)
    """
    ash_rgb = get_rgb(img/255.0)
    gauss_fimg = gaussian_filter_normalization(img)
    # Extract relevant bands
    band10 = gauss_fimg[2, :, :] # Band 10 (7.3 µm)
    band13 = gauss_fimg[5, :, :] # Band 13 (10.3 µm)

    # Compute Temperature Difference (TD)
    TD = band13 - band10

    # Compute Temperature Difference Index (TDi)
    TDi = TD / (1 + torch.exp(band13 - T_threshold))

    # Add TD and TDi as new channels
    img_with_TD_TDi = torch.cat([ash_rgb, TD.unsqueeze(0), TDi.unsqueeze(0)], dim=0)

    return img_with_TD_TDi
    


#####################################################################

########################## Visualization ##################################
def plot_rgb_and_mask(image_batch, mask_batch, pred_batch=None):
    batch_size = image_batch.shape[0]
    num_cols = 3 if pred_batch is None else 4

    fig, axes = plt.subplots(batch_size, num_cols, figsize=(10, 4 * batch_size))

    for i in range(batch_size):
        rgb_image = get_rgb(image_batch[i]).cpu().numpy().transpose((1, 2, 0))
        mask_band = mask_batch[i, 0, :, :].cpu().numpy()

        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Image {i + 1}")

        axes[i, 1].imshow(mask_band, interpolation="none")
        axes[i, 1].set_title(f"Mask {i + 1}")

        overlay = np.zeros_like(rgb_image)
        overlay[mask_band > 0.5] = 1.0
        axes[i, 2].imshow(rgb_image)
        axes[i, 2].imshow(overlay, cmap="cividis", alpha=0.4, interpolation="none")
        axes[i, 2].set_title(f"Combined {i + 1}")

        if pred_batch is not None:
            pred_band = pred_batch[i, 0, :, :].cpu().numpy()
            axes[i, 3].imshow(pred_band, interpolation="none")
            axes[i, 3].set_title(f"Prediction {i + 1}")

# Iterate over each batch and channel to plot histograms
def plot_batch_histogram(image):
    fig = plt.figure(figsize=(15, 15))
    for batch_idx in range(image.shape[0]):
        plt.subplot(image.shape[0], image.shape[1] + 1, batch_idx * (image.shape[1] + 1) + 1)
        plt.imshow(image[batch_idx, 0].cpu().numpy())
        plt.axis('off')
        plt.title(f'Batch {batch_idx + 1}')

        for channel_idx in range(image.shape[1]):
            plt.subplot(image.shape[0], image.shape[1] + 1, batch_idx * (image.shape[1] + 1) + channel_idx + 2)
            plt.hist(image[batch_idx, channel_idx].view(-1).cpu().numpy(), bins=50, color='blue', alpha=0.7)
            plt.title(f'Channel {channel_idx + 8}')

    plt.tight_layout()
    plt.show()
    
    
# Plot Batch Image and Mask 
def plot_image_and_mask(image_batch, mask_batch, pred_batch=None):
    batch_size = image_batch.shape[0]
    num_cols = 3 if pred_batch is None else 4

    fig, axes = plt.subplots(batch_size, num_cols, figsize=(10, 4 * batch_size))

    for i in range(batch_size):
        rgb_image = image_batch[i, 0:3, : , :].cpu().numpy().transpose((1, 2, 0))
        mask_band = mask_batch[i, 0, :, :].cpu().numpy()

        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Image {i + 1}")

        axes[i, 1].imshow(mask_band, interpolation="none")
        axes[i, 1].set_title(f"Mask {i + 1}")

        overlay = np.zeros_like(rgb_image)
        overlay[mask_band > 0.5] = 1.0
        axes[i, 2].imshow(rgb_image)
        axes[i, 2].imshow(overlay, cmap="cividis", alpha=0.4, interpolation="none")
        axes[i, 2].set_title(f"Overlay {i + 1}")

        if pred_batch is not None:
            pred_band = pred_batch[i, 0, :, :].cpu().numpy()
            axes[i, 3].imshow(pred_band, interpolation="none")
            axes[i, 3].set_title(f"Prediction {i + 1}")
            
# Plot xarray dataset 
def plot_dataset_band_hist(dataset, title):
    # Initialize a seaborn plot
    plt.figure(figsize=(10, 6))
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Define a sequential colormap
    cmap = plt.cm.get_cmap('tab20b', len(dataset.data_vars))

    # Create a KDE plot for each variable with sequential coloring
    for i, var_name in enumerate(dataset.data_vars):
        sns.kdeplot(dataset[var_name].values.flatten(), linewidth=2, label=var_name, color=cmap(i))

    # Add legend
    plt.legend()

    # Set labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(title)

    # Show plot
    plt.show()

# Box plots for ABI bands xarray dataset
def plot_bands_boxplots(dataset, title):
    # Define a colormap
    cmap = plt.cm.get_cmap('tab20b', len(dataset.data_vars))

    # Initialize a figure
    plt.figure(figsize=(12, 6))
    
    # Extract variable values and plot box plots for each variable
    for i, var_name in enumerate(dataset.data_vars):
        plt.boxplot(dataset[var_name].values.flatten(), 
                    positions=[int(var_name.split('_')[1])],
                    patch_artist=True,
                    boxprops=dict(facecolor=cmap(i)))

    # Set title and labels
    plt.title(title) #'Bands Box Plots of All Data'
    plt.xlabel('Bands')
    plt.ylabel('Values')

    # Show plot
    plt.show()
    
############################## Accuracy Matrix #########################

## Loss function
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        smooth = 1.0
        inputs = torch.sigmoid(inputs)
        true_pos = torch.sum(targets * inputs)
        false_neg = torch.sum(targets * (1 - inputs))
        false_pos = torch.sum((1 - targets) * inputs)
        return 1 - (true_pos + smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + smooth)

def binary_iou_dice_score(
    y_pred, y_true, metric="iou", apply_sigmoid=True, threshold=0.5, eps=1e-7
):
    assert metric in {"iou", "dice"}
    assert y_pred.shape == y_true.shape

    if apply_sigmoid:
        y_pred = torch.sigmoid(y_pred)

    y_pred = (y_pred > threshold).type(y_true.dtype)

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if metric == "iou":
        score = (intersection + eps) / (cardinality - intersection + eps)
    else:
        score = (2.0 * intersection + eps) / (cardinality + eps)

    return score

# Function to calculate metrics
def calculate_acc_metrics(preds, labels):
    flat_preds = preds.flatten()
    flat_labels = labels.flatten()

    accuracy = accuracy_score(flat_labels, flat_preds)
    precision = precision_score(flat_labels, flat_preds, zero_division=1)
    recall = recall_score(flat_labels, flat_preds, zero_division=1)
    f1 = f1_score(flat_labels, flat_preds, zero_division=1)
    iou = jaccard_score(flat_labels, flat_preds)

    return accuracy, precision, recall, f1, iou

## Function to calculate model accuracy agg matrics
def get_post_acc_matrix(pred_mask, actual_mask):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(pred_mask.flatten(), actual_mask.flatten()).ravel()
    total_samples = tn + fp + fn + tp
    confusion_matrix = [tn/total_samples, fn/total_samples, tp/total_samples, fp/total_samples]
    # Class-specific accuracy (for binary classes: class 0 and class 1)
    class_accuracy_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    class_accuracy_1 = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Error metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    return class_accuracy_0, class_accuracy_1, false_positive_rate, false_negative_rate, confusion_matrix 

# Plot for prediction mask Vs actual mask 
def visualize_results_with_overlay(preds, labels, n_samples=5, alpha=0.5):
    """
    Visualize actual and predicted masks on the same image with different colors.

    Parameters:
        preds (list or ndarray): List or array of predicted masks.
        labels (list or ndarray): List or array of ground truth masks.
        n_samples (int): Number of samples to visualize.
        alpha (float): Transparency level for overlaying the masks.
    """
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 5, 10))
    if n_samples == 1:
        axes = [axes]  # Ensure axes is iterable for a single sample case

    # Set a single title for the whole figure
    fig.suptitle('Overlay of Predicted and Ground Truth Masks:Red - Predicted, Green - Ground Truth', fontsize=16)

    for i in range(n_samples):
        pred_mask = preds[i][0]  # Assuming mask has a single channel
        true_mask = labels[i][0]

        # Create color masks
        pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.float32)
        true_colored = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.float32)

        # Color the predicted mask in red (R=1)
        pred_colored[:, :, 0] = pred_mask

        # Color the true mask in green (G=1)
        true_colored[:, :, 1] = true_mask

        # Combine the masks with different colors
        overlay = np.clip(alpha * pred_colored + alpha * true_colored, 0, 1)

        # Display the result
        axes[i].imshow(overlay)
        axes[i].set_title(f'Sample {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
# Plot model roc-auc plot
def plot_roc_auc(true_labels, pred_probs):
    """
    Plot ROC-AUC curve.

    Parameters:
        true_labels (numpy.array): Ground truth binary labels (flattened).
        pred_probs (numpy.array): Predicted probabilities for the positive class (flattened).
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()