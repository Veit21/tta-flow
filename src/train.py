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

from itertools import islice
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.data_loader import FlowMatchingTrainDataset, load_data_from_csv, get_transforms
from util import model_util, losses, flow_matching_util

# TODO: Tidy code!
#----------------------------------------------------------------------------
# Entry point to the training

@hydra.main(version_base=None, config_path="../preferences", config_name="config")
def main(cfg: DictConfig) -> None:

    # --- Options ---
    torch.manual_seed(cfg["train_param"]["seed"])
    np.random.seed(cfg["train_param"]["seed"])
    log                 = logging.getLogger(__name__)
    log_dir             = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    checkpoint_dir      = os.path.join(log_dir, "checkpoints")
    save_dir_val        = os.path.join(log_dir, "val")
    save_dir_train      = os.path.join(log_dir, "train")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir_val, "im"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_val, "gif"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_train, "im"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_train, "gif"), exist_ok=True)
    writer              = SummaryWriter(log_dir=log_dir)
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device {device}")

    # Get the transforms
    train_transforms, _, _  = get_transforms(config=cfg)

    # Load the data
    train_dataframe     = load_data_from_csv(csv_path=cfg["data"]["train_data_dir"])
    train_dataset       = FlowMatchingTrainDataset(
        dataframe=train_dataframe,
        transform=train_transforms,
    )

    train_dataloader    = DataLoader(dataset=train_dataset, batch_size=cfg["train_param"]["train_batch_size"], shuffle=True, pin_memory=True, drop_last=True)

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
    if cfg["train_param"]["continue_train"]:
        checkpoint_path = cfg["train_param"]["checkpoint_dir"]
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            log.warning(f"Checkpoint directory not found at {checkpoint_path}. Starting from scratch.")
        else:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
            if not checkpoints:
                log.warning(f"No checkpoints found in {checkpoint_path}. Starting from scratch.")
            else:
                # Extract step numbers and find latest
                steps = [int(f.split('_step_')[-1].replace('.pt', '')) for f in checkpoints]
                latest_idx = steps.index(max(steps))
                latest_checkpoint = os.path.join(checkpoint_path, checkpoints[latest_idx])
                
                log.info(f"Loading checkpoint from {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint)
                
                # Load model weights and check for incompatibilities
                net_missing, net_unexpected = net_model.load_state_dict(checkpoint['net_model'], strict=False)
                if net_missing:
                    log.warning(f"Missing keys in net_model: {net_missing}")
                if net_unexpected:
                    log.warning(f"Unexpected keys in net_model: {net_unexpected}")
                
                ema_missing, ema_unexpected = ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
                if ema_missing:
                    log.warning(f"Missing keys in ema_model: {ema_missing}")
                if ema_unexpected:
                    log.warning(f"Unexpected keys in ema_model: {ema_unexpected}")
                
                # Load optimizer and scheduler states
                optim.load_state_dict(checkpoint['optim'])
                sched.load_state_dict(checkpoint['sched'])
                
                # Update starting step
                start_step = checkpoint['step'] + 1
                log.info(f"Resuming training from step {start_step}")

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
    for step in tqdm(range(start_step, cfg["train_param"]["max_iterations"])):
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
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), cfg["train_param"]["grad_clip"])
        optim.step()
        sched.step()
        ema(source=net_model, target=ema_model, decay=cfg["train_param"]["ema_decay"])


        # Sample and Save the weights
        if (cfg["train_param"]["eval_num_steps"] > 0 and step % cfg["train_param"]["eval_num_steps"] == 0) and step != 0:

            # Save weights
            torch.save(
                {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                },
                os.path.join(checkpoint_dir, f"{cfg['model']['q_z']}_maestro-spectralis_weights_step_{step}.pt"),
            )

            # TODO: Overwork this part. I.e., do not use a dataloader here?
            # Validation loops
            generate_samples(cfg, model=net_model, eval_dataloader=val_dataloader, savedir_samples=save_dir_val, step=step, net_="normal", device=device)
            generate_samples(cfg, model=ema_model, eval_dataloader=val_dataloader, savedir_samples=save_dir_val, step=step, net_="ema", device=device)
            
    return 0


#----------------------------------------------------------------------------
# Helper functions

def generate_samples(cfg, model: nn.Module, eval_dataloader: DataLoader, savedir_samples: Path, step: int, net_: str, device):
    
    # Set model to evaluation mode
    model.eval()
    model_  = copy.deepcopy(model)

    # Cache all trajectories
    trajectories_cached = []
    x1_cached = []

    # Instantiate the ODE solver
    node_   = flow_matching_util.ODESolver(
        model=model_, 
        solver=cfg["train_param"]["solver"], 
        conditional=cfg["train_param"]["guided"], 
        sample_x=cfg["loss"]["target"] == "x"
    )

    # Iterate the eval_dataloader
    print(f"INFO: Validation - net {net_}")
    for batch in tqdm(islice(eval_dataloader, 9), total=9):
        
        # Draw batches of samples from the source domain
        x0, x1, y   = batch
        x0          = x0.to(device)[:1]     # Always just choose the first element of the batch for checking inference on training data # TODO: Remove this.
        x1          = x1.to(device)[:1]     # See above

        x0      = torch.randn_like(x1, device=device)   # Start from gaussian noise and generate a high-res sample

        with torch.no_grad():
            
            # ODE solve
            traj = node_(
                x       = x0,
                t_span  = torch.linspace(0, 1, 101, device=device), # TODO: Make num steos configurable
                y       = y
            )
            every_nth = 10      # Which nth samples from the trajectory to save later

        traj = traj.clip(-1, 1)         # Make sure images are in [-1, 1]
        traj = traj / 2 + 0.5           # Cast back to [0, 1]
        
        x1 = x1.clip(-1, 1)
        x1 = x1 / 2 + 0.5
    
        # Save
        trajectories_cached.append(traj)
        x1_cached.append(x1)
    
    # Save a subset of samples: x0, x1 and a GIF x0 -> x1
    save_samples(
        tensor_list=trajectories_cached,
        save_path=savedir_samples,
        step=step,
        net_=net_,
        grid_size=3,
        num_samples=9,  # TODO: Redundant
        every_nth=every_nth
    )

    # Activate train mode again
    model.train()


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

# TODO: Put into some utils class/file
def save_samples(tensor_list: list, save_path: str, step: str, net_: str, grid_size: int=8, num_samples: int=64, padding: int=2, every_nth: int=10):
    """
    Randomly samples sequences from a list of image tensors, saves initial and final
    frames as grids, and exports the full sequences as an animated GIF.

    Args:
        tensor_list (list): List of tensors, each shaped [T, 1, 1, H, W], where T is the number of frames.
        save_path (Path): Base directory where images and GIFs will be saved (subdirs 'im' and 'gif' are expected).
        step (str): Training step or identifier used in filenames.
        net_ (str): Network name or identifier used in filenames.
        grid_size (int, optional): Number of images per row/column in the grid. Defaults to 8.
        num_samples (int, optional): Number of sequences to sample for visualization. Defaults to 64.
        padding (int, optional): Padding (in pixels) between grid images. Defaults to 2.
        every_nth (int, optional): Which nth samples from a trajectory to save later. Defaults to 10.

    Raises:
        ValueError: If `tensor_list` contains fewer than `num_samples` elements.
    """

    # Sanity check
    if len(tensor_list) < num_samples:
        raise ValueError(f"Not enough tensors: need {num_samples}, got {len(tensor_list)}.")
    
    # Randomly choose tensors
    # sampled_tensors = random.sample(tensor_list, num_samples)
    indices = random.sample(range(len(tensor_list)), num_samples)
    sampled_tensors = [tensor_list[i] for i in indices]                             # List N x [100, 1, 1, H, W]

    # Get tensors at initial and final time step
    tensor_list_initial = [x[0, :] for x in sampled_tensors]        # Shape [1, 1, H, W]
    tensor_list_final   = [x[-1, :] for x in sampled_tensors]       # Shape [1, 1, H, W]


    # Stack into single tensors
    batch_initial   = torch.cat(tensor_list_initial)                                            # Shape [N, C, H, W]
    batch_final     = torch.cat(tensor_list_final)                                              # Shape [N, C, H, W]
    batch_T         = torch.cat(sampled_tensors, dim=1)           # Shape [100, N, 1, H, W], assume all tensors have the same shape
    T               = batch_T.shape[0]
    N               = batch_T.shape[1]

    # --- get random trajectory of batch_T ---
    rand_idx    = random.randint(0, N - 1)
    traj        = batch_T[:, rand_idx]          # Shape: [100, 1, H, W]
    traj_10     = traj[::every_nth]             # Shape: [11, 1, H, W], get every nth sample, typically 10th or 100th

    # --- save images ---
    # Create grid
    grid_initial    = make_grid(tensor=batch_initial, nrow=grid_size, padding=padding)
    grid_final      = make_grid(tensor=batch_final, nrow=grid_size, padding=padding)
    grid_traj       = make_grid(tensor=traj_10, nrow=10, padding=padding)

    
    # Convert to PIL Image and save
    to_pil          = ToPILImage()
    img_initial     = to_pil(grid_initial)
    img_final       = to_pil(grid_final)
    img_traj        = to_pil(grid_traj)
    img_initial.save(os.path.join(save_path, f"im/{step:06}_{net_}_x0.png"), format="PNG")
    img_final.save(os.path.join(save_path, f"im/{step:06}_{net_}_x1.png"), format="PNG")
    img_traj.save(os.path.join(save_path, f"im/{step:06}_{net_}_traj.png"), format="PNG")

    # --- save animation ---
    T = batch_T.shape[0]
    frames = []
    for t in range(T):          # Tensor at each time t has shape [N, 1, H, W]

        # Create grid
        grid    = make_grid(tensor=batch_T[t], nrow=grid_size, padding=padding)
        frames.append(to_pil(grid))
    
    # Save as GIF
    frames[0].save(
        os.path.join(save_path, f"gif/{step:06}_{net_}_im_grid.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )

    print(f"INFO: Samples saved.")


if __name__ == "__main__":

    # Start the main process.
    main()