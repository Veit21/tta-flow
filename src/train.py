###########################################################################
#
# Script for training the Flow Matching model.
#
###########################################################################

# Imports
import copy
import torch
import hydra
import logging
import os

import numpy as np
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.datasets import FlowMatchingTrainDataset, load_data_from_csv, get_transforms
from util import model_util, losses, flow_matching_util
from util.checkpoint_manager import CheckpointManager
from util.visualization import save_visualizations


#----------------------------------------------------------------------------
# Helper methods

def generate_samples(cfg: DictConfig, model: nn.Module, log_dir: Path, step: int, net: str, device: torch.device, logger: logging.Logger) -> None:
    """
    Generate validation samples using the trained model and save them to disk.
    
    Args:
        cfg (DictConfig): Configuration dictionary containing model and training parameters.
        model (nn.Module): Trained model to use for generation.
        savedir_samples (str): Directory path where samples will be saved.
        step (int): Current training step (used for filename).
        net (str): Network identifier ("normal" or "ema") for filename.
        device (torch.device): Device (CPU/GPU) to run generation on.
    """
    model.eval()
    ode_solver = flow_matching_util.ODESolver(
        model=copy.deepcopy(model), 
        solver=cfg["train_parameters"]["solver"], 
        sample_x=cfg["loss"]["target"] == "x"
    )
    
    # Configuration
    h, w            = cfg["model"]["params"]["dim"][1], cfg["model"]["params"]["dim"][2]
    num_timesteps   = cfg["train_parameters"]["num_integration_steps"]
    
    trajectories = []
    for _ in tqdm(range(cfg["train_parameters"]["num_val_samples"]), desc="Generating validation samples"):

        # Initialize from random noise
        x0 = torch.randn(1, 1, h, w, device=device)
        
        with torch.no_grad():
            trajectory = ode_solver(
                x=x0,
                t_span=torch.linspace(0, 1, num_timesteps, device=device),
            )
        
        # Normalize to [0, 1] range
        trajectory = trajectory.clip(-1, 1)
        trajectory = trajectory / 2 + 0.5
        trajectories.append(trajectory)
    
    # Save visualization
    save_visualizations(
        trajectories=trajectories,
        log_dir=log_dir,
        step=step,
        net=net,
        num_samples=cfg["train_parameters"]["num_val_samples"],
        logger=logger,
    )
    
    model.train()

def build_dataloader(cfg: DictConfig, log: logging.Logger) -> DataLoader:
    """TODO: Write a proper docstring

    Args:
        cfg (DictConfig): _description_

    Returns:
        DataLoader: _description_
    """

    # Get the transforms
    train_transforms, _, _  = get_transforms(config=cfg)

    # Load the data
    train_dataframe     = load_data_from_csv(csv_path=cfg["data"]["data_dir"])
    train_dataset       = FlowMatchingTrainDataset(
        dataframe=train_dataframe,
        log=log,
        transform=train_transforms,
    )
    train_dataloader    = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["train_parameters"]["train_batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    return train_dataloader

def build_models(cfg: DictConfig, log: logging.Logger, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """TODO: Write proper docstring.

    Args:
        cfg (DictConfig): _description_
        log (logging.Logger): _description_
        device (torch.device): _description_

    Returns:
        tuple[nn.Module, nn.Module]: _description_
    """

    # Define models
    net_model   = model_util.build_model(config=cfg, logger=log, device=device)
    ema_model   = copy.deepcopy(net_model)

    # Info
    n_params    = sum(p.data.nelement() for p in net_model.parameters())
    log.info("Model params: %.2f M" % (n_params / 1024 / 1024))

    return net_model, ema_model

def build_optimizer(cfg: DictConfig, net_model: nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """TODO: Write proper docstring here.

    Args:
        cfg (DictConfig): _description_
        net_model (nn.Module): _description_

    Returns:
        tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]: _description_
    """

    # Warmup plan
    def warmup_lr(step):
        return min(step, cfg["optimizer"]["warmup"]) / cfg["optimizer"]["warmup"]

    # Define optimizers
    optim   = torch.optim.AdamW(params=net_model.parameters(), lr=cfg["optimizer"]["learning_rate"])
    sched   = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=warmup_lr)
    return optim, sched

def build_flow_matcher(cfg: DictConfig, log: logging.Logger) -> tuple[TargetConditionalFlowMatcher, losses.FlowMatchingRegressionLoss]:
    """TODO: Write a proper docstring here.

    Args:
        cfg (DictConfig): _description_
        log (logging.Logger): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        tuple[TargetConditionalFlowMatcher, losses.FlowMatchingRegressionLoss]: _description_
    """

    # Define Flow Matching framework and the conditioning q(z)
    sigma   = cfg["model"]["sigma"]
    if cfg["model"]["q_z"] == "gaussian":

        # Transforms a source sample x_0 from gaussian noise to a target sample x_1 from the data distribution
        flow_matcher    = TargetConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(f"Unknown model >> {cfg['model']['q_z']} <<, must be one of ['gaussian'].")

    # Get loss function
    if cfg["loss"]["name"] == "FlowMatchingRegression":

        # Standard regression loss for flow matching, see Lipman et al. 2022, "Flow Matching for Generative Modeling"
        loss_func   = losses.FlowMatchingRegressionLoss(logger=log, regression_target=cfg["loss"]["target"])
    else:
        raise NotImplementedError(f"Unkonwn loss function >> {cfg['loss']['name']} <<, must be one of ['FlowMatchingRegression'].")
    
    return flow_matcher, loss_func

#----------------------------------------------------------------------------
# Main method

@hydra.main(version_base=None, config_path="../preferences", config_name="config")
def main(cfg: DictConfig) -> None:

    # Setup environment 
    torch.manual_seed(cfg["train_parameters"]["seed"])
    np.random.seed(cfg["train_parameters"]["seed"])
    torch.cuda.manual_seed_all(cfg["train_parameters"]["seed"])
    log     = logging.getLogger(__name__)
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    writer  = SummaryWriter(log_dir=log_dir)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device {device}")
    
    # Initialize checkpoint manager
    checkpoint_manager  = CheckpointManager(log_dir=log_dir, logger=log)

    # Build dataloader
    train_dataloader    = build_dataloader(cfg=cfg, log=log)
    train_iter          = iter(train_dataloader)

    # Build models
    net_model, ema_model    = build_models(cfg=cfg, log=log, device=device)
    
    # Optimizer
    optim, sched    = build_optimizer(cfg=cfg, net_model=net_model)

    # Flow matcher and loss
    flow_matcher, loss_func = build_flow_matcher(cfg=cfg, log=log)

    # Load from checkpoint if continuing training
    start_step = 0
    if cfg["train_parameters"]["continue_train"]:
        start_step = checkpoint_manager.load_checkpoint(
            net_model=net_model,
            ema_model=ema_model,
            optimizer=optim,
            scheduler=sched,
        )

    # Iterate DataLoader
    for step in tqdm(range(start_step, cfg["train_parameters"]["max_iterations"])):
        optim.zero_grad()
                
        # Draw next batch, reset iterator at end of epoch
        try:
            x0, x1 = next(train_iter)
        except StopIteration:
            train_iter  = iter(train_dataloader)
            x0, x1     = next(train_iter)

        x0 = x0.to(device)
        x1 = x1.to(device)

        # Compute the loss. Class also performs prediction and estimation of the ground truth velocity field.
        loss    = loss_func(
            flow_matcher=flow_matcher,
            net=net_model,
            x0=x0,
            x1=x1,
        )
        writer.add_scalar("loss", loss, step)

        # Update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), cfg["train_parameters"]["grad_clip"])
        optim.step()
        sched.step()
        model_util.ema(source=net_model, target=ema_model, decay=cfg["train_parameters"]["ema_decay"])

        # Sample and Save the weights
        if (cfg["train_parameters"]["eval_num_steps"] > 0 and step % cfg["train_parameters"]["eval_num_steps"] == 0) and step != 0:

            # Save weights
            checkpoint_manager.save_checkpoint(
                net_model=net_model,
                ema_model=ema_model,
                optimizer=optim,
                scheduler=sched,
                step=step,
                model_name=f"{cfg['model']['q_z']}_{cfg["data"]["name"]}",
            )

            # Validation loops
            generate_samples(cfg, model=net_model, log_dir=log_dir, step=step, net="normal", device=device, logger=log)
            generate_samples(cfg, model=ema_model, log_dir=log_dir, step=step, net="ema", device=device, logger=log)


    # Close logging
    writer.close()
    return 0


if __name__ == "__main__":

    # Start the main process.
    main()