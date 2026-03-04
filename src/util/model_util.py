##############################################################
#
#       This script defines some helper functions related to the models
#
##############################################################

# Import libraries
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
    logger.info(f"Building model >> {config['model']['name']} << from {config['model']['module']}")

    # Load model class
    model_class = get_class(config['model']['module'])
    
    # Pass arguments
    model = model_class(**config['model']['params']).to(device)
    return model