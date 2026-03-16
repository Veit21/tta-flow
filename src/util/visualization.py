##############################################################
#
#   Helper functions for visualization
#
##############################################################

# Imports
import os
import logging
import random
import torch

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


def save_visualizations(
    trajectories: list,
    log_dir: str,
    step: int,
    net: str,
    logger: logging.Logger,
    grid_size: int = 3,
    num_samples: int = 9,
    padding: int = 2,
    every_nth: int = 10,
):
    """Saves trajectory visualizations as static image grids and animated GIFs.
    
    This function randomly samples a subset of the provided trajectories and generates
    three types of visualizations saved within the validation directory (`<log_dir>/val/`):
    1. A static grid of the final generated frames (x1).
    2. A static grid showing the progression of a single denoising trajectory subsampled over time.
    3. An animated GIF showing the generation process of the grid across all timesteps.

    Args:
        trajectories (list): List of trajectory tensors, where each tensor typically has the shape [T, 1, 1, H, W].
        log_dir (str): Base logging directory. Visualizations are saved in subdirectories `val/im` and `val/gif`.
        step (int): The current training step or iteration, used for naming the output files.
        net (str): Identifier for the network (e.g., "ema" or "normal"), used for naming the output files.
        logger (logging.Logger): Logger object.
        grid_size (int, optional): Number of images per row in the final frame and GIF grids. Defaults to 3.
        num_samples (int, optional): Maximum number of trajectories to randomly select for visualization. Defaults to 9.
        padding (int, optional): Number of padding pixels between images in the generated grids. Defaults to 2.
        every_nth (int, optional): Step interval for subsampling the single trajectory progression grid. Defaults to 10.
    """
    
    # Ensure output directories exist
    save_path   = log_dir / "val"
    (save_path / "im").mkdir(parents=True, exist_ok=True)
    (save_path / "gif").mkdir(parents=True, exist_ok=True)
    
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