##############################################################
#
#   Define some helper functions for visualization
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