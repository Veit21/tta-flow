###########################################################################
#
# Script for training the Flow Matching model.
#
###########################################################################

# Imports
import random
import copy
import torch
import hydra
import logging
import os

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from omegaconf import DictConfig
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.datasets import FlowMatchingTrainDataset, load_data_from_csv, get_transforms
from util import model_util, losses, flow_matching_util
from util.checkpoint_manager import CheckpointManager

# TODO: Tidy code!
#----------------------------------------------------------------------------
# Entry point to the training

@hydra.main(version_base=None, config_path="../preferences", config_name="config")
def main(cfg: DictConfig) -> None:

    # --- Options ---
    torch.manual_seed(cfg["train_parameters"]["seed"])
    np.random.seed(cfg["train_parameters"]["seed"])
    log                 = logging.getLogger(__name__)
    log_dir             = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    checkpoint_dir      = os.path.join(log_dir, "checkpoints")
    save_dir_val        = os.path.join(log_dir, "val")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir_val, "im"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_val, "gif"), exist_ok=True)
    writer              = SummaryWriter(log_dir=log_dir)
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device {device}")
    
    # Initialize checkpoint manager
    checkpoint_manager  = CheckpointManager(checkpoint_dir=checkpoint_dir, logger=log)

    # Get the transforms
    train_transforms, _, _  = get_transforms(config=cfg)

    # Load the data
    train_dataframe     = load_data_from_csv(csv_path=cfg["data"]["data_dir"])
    train_dataset       = FlowMatchingTrainDataset(
        dataframe=train_dataframe,
        transform=train_transforms,
    )
    train_dataloader    = DataLoader(dataset=train_dataset, batch_size=cfg["train_parameters"]["train_batch_size"], shuffle=True, pin_memory=True, drop_last=True)

    # Define the models
    net_model   = model_util.build_model(config=cfg, logger=log, device=device)
    ema_model   = copy.deepcopy(net_model)

    # Warmup plan
    def warmup_lr(step):
        return min(step, cfg["optimizer"]["warmup"]) / cfg["optimizer"]["warmup"]

    # Define optimizers
    optim       = torch.optim.AdamW(params=net_model.parameters(), lr=cfg["optimizer"]["learning_rate"])
    sched       = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=warmup_lr)

    # Show model size
    model_size  = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    log.info("Model params: %.2f M" % (model_size / 1024 / 1024))

    # Load from checkpoint if continuing training
    start_step = 0
    if cfg["train_parameters"]["continue_train"]:
        start_step = checkpoint_manager.load_checkpoint(
            net_model=net_model,
            ema_model=ema_model,
            optimizer=optim,
            scheduler=sched,
        )

    # Define Flow Matching framework and the conditioning q(z)
    sigma = cfg["model"]["sigma"]
    if cfg["model"]["q_z"] == "gaussian":

        # Transforms a source sample x_0 from gaussian noise to a target sample x_1 from the data distribution
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(f"Unknown model >> {cfg['model']['q_z']} <<, must be one of ['gaussian'].")

    # Get the loss function
    if cfg["loss"]["name"] == "FlowMatchingRegression":

        # Standard regression loss for flow matching, see Lipman et al. 2022, "Flow Matching for Generative Modeling"
        loss_func   = losses.FlowMatchingRegressionLoss(logger=log, regression_target=cfg["loss"]["target"])
    else:
        raise NotImplementedError(f"Unkonwn loss function >> {cfg['loss']['name']} <<, must be one of ['FlowMatchingRegression'].")

    # Iterate DataLoader
    for step in tqdm(range(start_step, cfg["train_parameters"]["max_iterations"])):
        optim.zero_grad()
                
        # Independently draw samples from p0 and p1
        batch       = next(iter(train_dataloader))
        x0, x1      = batch
        x0          = x0.to(device)
        x1          = x1.to(device)

        # Compute the loss. Class also performs prediction and estimation of the ground truth velocity field.
        loss    = loss_func(
            flow_matcher=FM,
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
        ema(source=net_model, target=ema_model, decay=cfg["train_parameters"]["ema_decay"])

        # Sample and Save the weights
        if (cfg["train_parameters"]["eval_num_steps"] > 0 and step % cfg["train_parameters"]["eval_num_steps"] == 0) and step != 0:

            # Save weights
            checkpoint_manager.save_checkpoint(
                net_model=net_model,
                ema_model=ema_model,
                optimizer=optim,
                scheduler=sched,
                step=step,
                model_name=f"{cfg['model']['q_z']}_maestro-spectralis",
            )

            # Validation loops
            generate_samples(cfg, model=net_model, savedir_samples=save_dir_val, step=step, net="normal", device=device, logger=log)
            generate_samples(cfg, model=ema_model, savedir_samples=save_dir_val, step=step, net="ema", device=device, logger=log)
            
    return 0


#----------------------------------------------------------------------------
# Helper functions

def generate_samples(cfg: DictConfig, model: nn.Module, savedir_samples: str, step: int, net: str, device: torch.device, logger: logging.Logger) -> None:
    """
    Generate validation samples using the trained model and save them to disk.
    
    Args:
        cfg (DictConfig): Configuration dictionary containing model and training parameters.
        model (nn.Module): Trained model to use for generation.
        savedir_samples (str): Directory path where samples will be saved.
        step (int): Current training step (used for filename).
        net_ (str): Network identifier ("normal" or "ema") for filename.
        device (torch.device): Device (CPU/GPU) to run generation on.
    """
    model.eval()
    ode_solver = flow_matching_util.ODESolver(
        model=copy.deepcopy(model), 
        solver=cfg["train_parameters"]["solver"], 
        sample_x=cfg["loss"]["target"] == "x"
    )
    
    # Configuration
    h, w = cfg["model"]["params"]["dim"][1], cfg["model"]["params"]["dim"][2]
    num_timesteps = 101
    
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
    _save_visualizations(
        trajectories=trajectories,
        save_path=savedir_samples,
        step=step,
        net=net,
        num_samples=cfg["train_parameters"]["num_val_samples"],
        logger=logger,
    )
    
    model.train()

def _save_visualizations(
    trajectories: list,
    save_path: str,
    step: int,
    net: str,
    logger: logging.Logger,
    grid_size: int = 3,
    num_samples: int = 9,
    padding: int = 2,
    every_nth: int = 10,
) -> None:
    """
    Save trajectory visualizations as static images and animated GIFs.
    
    Creates three types of visualizations:
    - Grid of initial frames (x0)
    - Grid of final frames (x1)
    - Grid of sampled trajectory steps and animated GIF
    """
    
    # Ensure output directories exist
    os.makedirs(os.path.join(save_path, "im"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "gif"), exist_ok=True)
    
    # Select subset of trajectories
    actual_num_samples = min(len(trajectories), num_samples)
    selected_indices = random.sample(range(len(trajectories)), actual_num_samples)
    selected_trajectories = [trajectories[i] for i in selected_indices]
    
    # Extracted trajectories shape: [T, 1, 1, H, W] -> [T, 1, H, W]
    squeezed_trajectories = [traj.squeeze(2) for traj in selected_trajectories]
    
    # Get final frames: take the last timestep of each squeezed trajectory
    # traj[-1] shape is [1, H, W]. Stack them along dim=0 to get [N, 1, H, W]
    final_frames = torch.stack([traj[-1] for traj in squeezed_trajectories], dim=0)
    
    # Stack all time steps along the batch dimension: [T, N, 1, H, W]
    stacked = torch.stack(squeezed_trajectories, dim=1)
    num_timesteps = stacked.shape[0]
    
    to_pil = ToPILImage()
    
    # Save final frames (x1) static grid
    grid = make_grid(final_frames, nrow=grid_size, padding=padding)
    img = to_pil(grid)
    img.save(os.path.join(save_path, f"im/{step:06}_{net}_x1.png"), format="PNG")
    
    # Save sampled trajectory
    sample_idx = random.randint(0, actual_num_samples - 1)
    trajectory_sample = stacked[:, sample_idx]              # Shape: [T, 1, H, W]
    trajectory_subsampled = trajectory_sample[::every_nth]  # Shape: [T', 1, H, W]
    
    grid_traj = make_grid(trajectory_subsampled, nrow=10, padding=padding)
    to_pil(grid_traj).save(
        os.path.join(save_path, f"im/{step:06}_{net}_traj.png"),
        format="PNG",
    )
    
    # Save animated GIF of all timesteps
    # stacked[t] is already [N, 1, H, W], perfectly sized for make_grid.
    frames_gif = [
        to_pil(make_grid(stacked[t], nrow=grid_size, padding=padding))
        for t in range(num_timesteps)
    ]
    
    frames_gif[0].save(
        os.path.join(save_path, f"gif/{step:06}_{net}_im_grid.gif"),
        save_all=True,
        append_images=frames_gif[1:],
        duration=100,
        loop=0,
    )
    
    logger.info(f"Visualizations saved: step {step}, network {net}")

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


if __name__ == "__main__":

    # Start the main process.
    main()