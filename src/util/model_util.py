##############################################################
#
#   This script defines some helper functions related to the models
#
##############################################################

# Import libraries
import torch.nn as nn

from hydra.utils import get_class
from omegaconf import DictConfig

def build_model(config: DictConfig, logger, device) -> None:
    """
    Instantiates a model defined by the config file.

    Args:
        config (DictConfig): Config (created from some .yaml file) which defines the model to load.
        logger (Logger): Logs console output.
        device (str): Device to load the model to. 'gpu' or 'cpu'.

    Returns:
        Model: Instantiated model object.
    """

    # Log info
    if logger:
        logger.info(f"Building model >> {config['model']['name']} << from {config['model']['module']}")

    # Load model class
    model_class = get_class(config['model']['module'])
    
    # Pass arguments
    model = model_class(**config['model']['params']).to(device)
    return model

def ema(source: nn.Module, target: nn.Module, decay: float):
    """Exponential moving average decay of the model parameters.

    Args:
        source (nn.Module): Source network to read the current weights from.
        target (nn.Module): Target network whose weights should be decayed.
        decay (float): Decay rate of the model parameters.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )