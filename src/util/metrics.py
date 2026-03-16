###########################################################################
#
# This script contains functions to calculate evaluation metrics
#
###########################################################################

# Imports
import torch

import numpy as np

from tqdm import tqdm
from scipy import linalg
from ..models.retfound_model import RETFound_mae


def calculate_fid(act1: np.ndarray, act2: np.ndarray, eps: float=1e-6) -> float:
    """Calculates the Frechet 'Inception' Distance (FID) between two sets of activations.
    NOTE: The embeddings might actually not be implemented using an Inception network, so the name can be misleading.

    Args:
        act1 (np.ndarray): Batch of activations.
        act2 (np.ndarray): Batch of activations.
        eps (float, optional): Small value to avoid singularity. Defaults to 1e-6.

    Raises:
        ValueError: If the product of covariance matrices is singular.

    Returns:
        float: The FID score.
    """

    mu1 = np.mean(act1, axis=0)
    mu2 = np.mean(act2, axis=0)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1  = np.cov(act1, rowvar=False)
    sigma2  = np.cov(act2, rowvar=False)

    sigma1  = np.atleast_2d(sigma1)
    sigma2  = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_retfound_activations_from_array(images: np.ndarray, device: torch.device, batch_size: int = 50) -> np.ndarray:
    """Computes the activations from a numpy array of images using RETFound as feature extractor.
    
    Args:
        images (np.ndarray): Array of shape (N, 512, 512) with values in [0, 1]
        device (torch.device): Device to run calculations on.
        batch_size (int, optional): Batch size of images for the model to process at once. Defaults to 50.
    
    Returns:
        np.ndarray: A numpy array of dimension (num_images, embedding_dim) that contains the activations.
    """
    N, H, W = images.shape
    
    if batch_size > N:
        print(f"Warning: batch size ({batch_size}) is bigger than the data size ({N}). Setting batch size to data size")
        batch_size = N
    
    # Load RETFound model
    model = RETFound_mae()
    model.to(device)
    model.eval()
    
    # Convert to torch tensor and add channel dimension
    # Shape: (N, H, W) -> (N, 1, H, W)
    images_tensor = torch.from_numpy(images).float().unsqueeze(1)
    
    # Convert grayscale to RGB by repeating channels
    # Shape: (N, 1, H, W) -> (N, 3, H, W)
    images_tensor = images_tensor.repeat(1, 3, 1, 1)
    
    # Create TensorDataset and DataLoader
    dataset     = torch.utils.data.TensorDataset(images_tensor)
    dataloader  = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Collect predictions
    pred_list = []
    
    # Iterate the dataset
    for (batch,) in tqdm(dataloader, desc="Computing embeddings"):
        batch = batch.to(device)  # Shape: (B, 3, H, W)
        
        with torch.no_grad():
            pred = model(batch)     # Reshapes to (B, 3, 224, 224) && (batch - batch.mean) / batch.std as normalization
        
        pred = pred.squeeze().cpu().numpy()
        
        # Handle single-sample batches (squeeze removes all dims of size 1)
        if pred.ndim == 1 and batch.shape[0] == 1:
            pred = pred[np.newaxis, :]
        
        pred_list.append(pred)
    
    # Concatenate all predictions in batch dimension
    pred_arr = np.concatenate(pred_list, axis=0)
    
    return pred_arr