###########################################################################
#
# This script contains functions to calculate some metrics on the results
#
###########################################################################

# Imports
import torch
import timm

import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from scipy import linalg
from models.inception_model import InceptionV3
from models.retfound_model import RETFound_mae
from models.mirage_model import MIRAGEWrapper
from util import datasets


def calculate_fid(act1: np.ndarray, act2: np.ndarray, eps: float=1e-6) -> float:
    """Calculates the Frechet Inception Distance (FID) between two sets of activations.

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

def get_activations_from_array(images: np.ndarray, device: torch.device, batch_size: int = 50) -> np.ndarray:
    """
    Computes the activations from a numpy array of images using RETFound as feature extractor.
    
    Args:
        images (np.ndarray): Array of shape (N, 512, 512) with values in [0, 1]
        device (torch.device): Device to run calculations on.
        batch_size (int, optional): Batch size of images for the model to process at once. Defaults to 50.
    
    Returns:
        np.ndarray: A numpy array of dimension (num_images, dims) that contains the activations.
    """
    N, H, W = images.shape
    assert H == 512 and W == 512, f"Expected images of size 512x512, got {H}x{W}"
    
    if batch_size > N:
        print(f"Warning: batch size ({batch_size}) is bigger than the data size ({N}). Setting batch size to data size")
        batch_size = N
    
    # Load RETFound model
    model = RETFound_mae()
    model.to(device)
    model.eval()
    
    # Convert to torch tensor and add channel dimension
    # Shape: (N, 512, 512) -> (N, 1, 512, 512)
    images_tensor = torch.from_numpy(images).float().unsqueeze(1)
    
    # Convert grayscale to RGB by repeating channels
    # Shape: (N, 1, 512, 512) -> (N, 3, 512, 512)
    images_tensor = images_tensor.repeat(1, 3, 1, 1)
    
    # Normalize to mean=0, std=1
    # Calculate mean and std across all pixels (keeping batch and channel dims)
    mean = images_tensor.mean(dim=[2, 3], keepdim=True)
    std = images_tensor.std(dim=[2, 3], keepdim=True)
    images_tensor = (images_tensor - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
    
    # Create TensorDataset and DataLoader
    dataset = torch.utils.data.TensorDataset(images_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Collect predictions
    pred_list = []
    
    # Iterate the dataset
    for (batch,) in tqdm(dataloader, desc="Computing embeddings"):
        batch = batch.to(device)  # Shape: (B, 3, 512, 512)
        
        with torch.no_grad():
            pred = model(batch)
        
        pred = pred.squeeze().cpu().numpy()
        
        # Handle single-sample batches (squeeze removes all dims of size 1)
        if pred.ndim == 1 and batch.shape[0] == 1:
            pred = pred[np.newaxis, :]
        
        pred_list.append(pred)
    
    # Concatenate all predictions in batch dimension
    pred_arr = np.concatenate(pred_list, axis=0)
    
    return pred_arr


def get_activations_from_path(files: list, device: torch.device, model_name: str="inception", load_from_numpy: bool=True, batch_size: int=50) -> np.ndarray:
    """Computes the activations from a list of images given a specific model which serves as feature extractor.

    Args:
        files (list): List of image file paths.
        device (torch.device): Device to run calculations on.
        model_name (str, optional): Defines which model to load as a feature extrator. Must be in list ["inception", "retfound", "mirage", "dino"]. Defaults to "inception".
        load_from_numpy (bool, optoinal): Set True if the data to load is saved as .npy. False otherwise. Defaults to False.
        batch_size (int, optional): Batch size of images for the model to process at once. Defaults to 50.

    Returns:
        np.ndarray: A numpy array of dimension (num_images, dims) that contains the activations of the given tensor when providing the model with the query tensor.
    """

    if batch_size > len(files):
        print("Warning: batch size is bigger than the data size. Setting batch size to data size")
        batch_size = len(files)
    
    # TODO: Put this intransparent data preprocessing from every class here to the model definition!
    # Data pre-processing   NOTE: Data needs to be in [0, 1]! Additionally, InceptionV3 normalizes to [-1, 1] and RETFound to 0 mean, 1 std internally.

    # Load model
    if model_name == "inception":
        model       = InceptionV3()

        # Define model specific transforms
        transforms  = A.Compose([
            A.Resize(height=512, width=512),
            A.ToTensorV2()
        ])
    elif model_name == "retfound":
        model       = RETFound_mae()

        # Define model specific transforms
        transforms  = A.Compose([
            A.Resize(height=512, width=512),
            A.ToTensorV2()
        ])
    elif model_name == "mirage":
        model       = MIRAGEWrapper(device=device)

        # Define model specific transforms
        transforms  = A.Compose([
            A.Resize(height=512, width=512),
            A.ToTensorV2()
        ])
    elif model_name == "dino":
        model       = timm.create_model(
            'timm/vit_large_patch16_dinov3.lvd1689m',
            pretrained=True,
            # features_only=True,
            num_classes=0,      # Remove nn.Linear classifier
        )
        
        # Define model specific transforms
        transforms  = A.Compose([
        A.Resize(height=256,width=256),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=1,
        ),
        A.ToTensorV2(),
    ])
    else:
        raise ValueError(f"The model corresponding to the name >> {model_name} << does not exist!")
    
    model.to(device)
    model.eval()

    # Construct DataLoader
    dataset     = datasets.Dataset_Cached(paths_list=files, is_3d=True, load_from_numpy=load_from_numpy, transform=transforms)
    dataloader  = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Collect predictions
    pred_list = []

    # Iterate the dataset
    for batch in tqdm(dataloader):
        assert batch.dim() == 4, f"Expected input to have 4 dimensions. Got shape: {batch.shape} instead."
        
        _b, _c, _h, _w  = batch.shape
        assert _c == 3, f"Expected input to have 3 channels. Got >> {_c} << instead."

        batch   = batch.to(device)                  # Image should have dimensions (B, C, H, W)

        with torch.no_grad():
            pred = model(batch)
        
        pred = pred.squeeze().cpu().numpy()
        pred_list.append(pred)

    # Concatenate all predictions in batch dimension
    pred_arr = np.concatenate(pred_list, axis=0)

    # TODO: How to make sure the embeddings are meaningful?
    return pred_arr