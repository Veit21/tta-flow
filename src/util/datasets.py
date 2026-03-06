# Import libraries
import logging
import torch
import numpy as np
import pandas as pd
import albumentations as A

from hydra.utils import get_class
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset


#----------------------------------------------------------------------------
# Dataset classes for loading and caching the data
 
class FlowMatchingTrainDataset(Dataset):
    """
    PyTorch Dataset for flow matching training that loads and caches 3D volumetric data.
    
    This dataset loads 3D volumes from disk, splits them into 2D slices, and applies
    augmentation transformations. For each sample, it returns a pair of images: a random
    noise tensor (x_0) and a transformed target image (x_1).
    
    Attributes:
        df (pd.DataFrame): DataFrame containing paths to volumetric data.
        transform: Albumentations augmentation pipeline to apply to images.
        volumes_tag (str): Column name in the DataFrame that contains volume file paths.
        cached_files (list): List of cached 2D image slices loaded from 3D volumes.
    """
    
    def __init__(self, dataframe: pd.DataFrame, log: logging.Logger, transform=None, volumes_tag="volume"):
        """
        Initialize the FlowMatchingTrainDataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with volume file paths.
            transform (albumentations.Compose, optional): Augmentation pipeline to apply to images.
                Defaults to None if no transformations are applied.
            volumes_tag (str, optional): Column name in dataframe containing volume paths.
                Defaults to "volume".
        """
        self.df             = dataframe
        self.log            = log
        self.transform      = transform
        self.volumes_tag    = volumes_tag

        # Cache data
        self.cached_files   = self._cache_data()

    def __len__(self):
        return len(self.cached_files)
    
    def __getitem__(self, idx):
        
        # Draw images from cache
        x_1 = self.cached_files[idx].transpose(1, 2, 0)     # (C, H, W) -> (H, W, C)

        # Apply transforms
        if self.transform:
            x_1 = self.transform(image=x_1)["image"]
            x_0 = torch.randn_like(x_1, dtype=torch.float32)
            
        return x_0, x_1

    def _cache_data(self) -> tuple:
        """
        Load 3D volumetric data from disk and split into 2D slices.
        
        This method reads all 3D volumes specified in the dataframe, splits each volume
        along the depth dimension into individual 2D slices, and caches them in memory.
        
        Returns:
            tuple: List of cached 2D image slices, each of shape (H, W) or (C, H, W).
        """
        cached_images    = []

        for path in tqdm(self.df[self.volumes_tag], desc="Loading files"):
            volume  = np.load(path).astype(np.float32)
            slices  = np.split(volume, volume.shape[0], axis=0)     # Split (D, H, W) in D x (1, H, W) arrays
            cached_images += slices

        self.log.info(f"Loaded {len(cached_images)} target images")
    
        return cached_images
    

class FlowMatchingInferenceDataset(Dataset):
    """
    PyTorch Dataset for inference that loads and caches full 3D volumetric data.
    
    Unlike the training dataset, this dataset loads complete 3D volumes without slicing.
    It optionally applies augmentation transformations slice-by-slice and returns the
    full 3D volume as a tensor.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing paths to volumetric data.
        transform: Albumentations augmentation pipeline to apply to each 2D slice.
        volumes_tag (str): Column name in the DataFrame that contains volume file paths.
        cached_volumes (list): List of cached full 3D volumes.
    """
    
    def __init__(self, dataframe: pd.DataFrame, log: logging.Logger, transform=None, volumes_tag="volume"):
        """
        Initialize the FlowMatchingInferenceDataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with volume file paths.
            transform (albumentations.Compose, optional): Augmentation pipeline to apply to each 2D slice.
                Defaults to None if no transformations are applied.
            volumes_tag (str, optional): Column name in dataframe containing volume paths.
                Defaults to "volume".
        """
        self.df             = dataframe
        self.log            = log
        self.transform      = transform
        self.volumes_tag    = volumes_tag
        
        # Cache data
        self.cached_volumes = self._cache_data()
    
    def __len__(self) -> int:
        return len(self.cached_volumes)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        volume = self.cached_volumes[idx].copy()  # (D, H, W)
        
        # Apply transforms
        if self.transform:
          volume = self.transform(volume=volume)["volume"]

        return volume
    
    def _cache_data(self) -> list:
        """
        Load 3D volumetric data from disk and cache them.
        
        This method reads all 3D volumes specified in the dataframe and caches them in memory.
        
        Returns:
            list: List of cached 3D volumes, each of shape (D, H, W).
        """
        cached_volumes = []
        
        for path in tqdm(self.df[self.volumes_tag], desc="Loading volumes"):
            volume = np.load(path).astype(np.float32)
            cached_volumes.append(volume)
        
        self.log.info(f"Loaded {len(cached_volumes)} volumes")
        
        return cached_volumes 


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads the content of a .csv file and returns it as a pd.DataFrame.

    Args:
        csv_path (str): Path to the .csv file that contains information about the data.

    Returns:
        pd.DataFrame: DataFrame object which contains the data.
    Note:
        Should this method implement more functionalities?
    """
    return pd.read_csv(csv_path)


def get_transforms(config: DictConfig) -> tuple:
    """
    Loads the augmentation pipeline from a .yaml file that is passed in the command line.

    Args:
        config (DictConfig): A config dictionary that contains the names and parameters for the augmentations to load. The dictionary is created from a .yaml file.

    Returns:
        tuple: A pair of train- and validation transform pipelines.
    """
    
    # List the augmentations
    train_aug_list  = []
    val_aug_list    = []
    test_aug_list   = []

    # Get train augmentations
    for t_train in config.transforms.train:
        cls = get_class(t_train.name)
        train_aug_list.append(cls(**t_train.params))

    # Get validation augmentations
    for t_val in config.transforms.validation:
        cls = get_class(t_val.name)
        val_aug_list.append(cls(**t_val.params))
    
    # Get test augmentations
    for t_test in config.transforms.test:
        cls = get_class(t_test.name)
        test_aug_list.append(cls(**t_test.params))

    # Compose pipelines
    train_transforms    = A.Compose(train_aug_list)
    val_transforms      = A.Compose(val_aug_list)
    test_transforms     = A.Compose(test_aug_list)

    return train_transforms, val_transforms, test_transforms
