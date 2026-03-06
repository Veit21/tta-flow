###########################################################################
#
# Script for inference.
#
###########################################################################

import logging
import torch
import hydra
import numpy as np
import torch.nn as nn

from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Adjust these imports to match your exact file structure
from util.datasets import FlowMatchingInferenceDataset, get_transforms, load_data_from_csv
from torch.utils.data import DataLoader
from util import model_util, flow_matching_util
from util.checkpoint_manager import CheckpointManager

#----------------------------------------------------------------------------
# Helper methods

def build_dataloader(cfg: DictConfig, log: logging.Logger) -> DataLoader:
    """TODO: Write a proper docstring

    Args:
        cfg (DictConfig): _description_
        log (loggin.Logger): _description_

    Returns:
        DataLoader: _description_
    """

    # Get the transforms
    _, _, test_transforms  = get_transforms(config=cfg)

    # Load the data
    test_dataframe     = load_data_from_csv(csv_path=cfg["data"]["data_dir"])
    test_dataset       = FlowMatchingInferenceDataset(
        dataframe=test_dataframe,
        log=log,
        transform=test_transforms,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataloader

def initialize_model(cfg: DictConfig, log: logging.Logger, device: torch.device, experiment_root_dir: Path) -> nn.Module:
    """TODO: Write proper docstring here.

    Args:
        cfg (DictConfig): _description_
        log (logging.Logger): _description_
        device (torch.device): _description_
        experiment_root_dir (Path): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        nn.Module: _description_
    """
    model               = model_util.build_model(config=cfg, logger=log, device=device)
    checkpoint_manager  = CheckpointManager(log_dir=experiment_root_dir, logger=log)
    ckpt_path           = Path(checkpoint_manager.find_latest_checkpoint())
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    # Load model weights
    log.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint['ema_model'], strict=False)
    if missing:
        log.warning(f"Missing keys in ema_model: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys in ema_model: {unexpected}")
    return model

def generate_reference_trajectories(cfg: DictConfig, log: logging.Logger, ode_solver: flow_matching_util.ODESolver, device: torch.device) -> np.ndarray:
    """TODO: Generate proper docstring here.

    Args:
        cfg (DictConfig): _description_
        log (logging.Logger): _description_
        ode_solver (flow_matching_util.ODESolver): _description_
        device (torch.device): _description_

    Returns:
        np.ndarray: _description_
    """
    log.info(f"Generating N={cfg["test_parameters"]["num_reference_trajectories"]} reference trajectories...")
    h, w = cfg["model"]["params"]["dim"][1], cfg["model"]["params"]["dim"][2]
    traj_reference_list = []

    # Iterate
    for _ in tqdm(range(cfg["test_parameters"]["num_reference_trajectories"]), desc="Iterating trials"):

        # Draw initial condition
        x0  = torch.randn((1, 1, h, w), device=device)

        # Generate
        with torch.no_grad():
            traj_reference = ode_solver(
                x       = x0,
                t_span  = torch.linspace(0, 1, cfg["test_parameters"]["num_integration_steps"], device=device),
            )
            traj_reference_np = traj_reference.detach().cpu().numpy()
            traj_reference_list.append(traj_reference_np)

    reference_trajectories_np = np.stack(traj_reference_list).astype(np.float32)
    return reference_trajectories_np

#----------------------------------------------------------------------------
# Main method

@hydra.main(version_base=None, config_path="../preferences", config_name="config")
def main(cfg: DictConfig):

    # Setup logging
    log = logging.getLogger(__name__)
    
    exp_dir = Path(cfg["experiment_path"])
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist.")
        
    # Set up I/O
    out_dir = exp_dir / "inference" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataloader
    test_dataloader = build_dataloader(cfg=cfg, log=log)

    # Initialize model
    model = initialize_model(cfg=cfg, log=log, device=device, experiment_root_dir=exp_dir)
    model.eval()

    # Initialize ODE solver
    ode_solver = flow_matching_util.ODESolver(
        model=deepcopy(model), 
        solver=cfg["train_parameters"]["solver"], 
        sample_x=cfg["loss"]["target"] == "x"
    )
    num_timesteps   = cfg["test_parameters"]["num_integration_steps"] 

    # Generate N reference trajectories
    num_references  = cfg["test_parameters"]["num_reference_trajectories"]
    s_target        = int(cfg["test_parameters"]["s_target"] * num_timesteps)

    # Validate _N and s_target are positive integer (incl. 0)
    if not isinstance(num_references, int):
        raise TypeError(f"N and s_target must be integer! Got: N={num_references} (type: {type(num_references).__name__})")
    if num_references > 0:
        reference_trajectories_np = generate_reference_trajectories(cfg=cfg, log=log, ode_solver=ode_solver, device=device)
    elif num_references == 0:
        log.info("Histogram matching deactivated!")
    else:
        raise ValueError(f"Only non-negative integer N allowed! Got: N={num_references}, Expected: N>=0")

    # Inference Loop
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader, desc="Processing Volumes")):
            
            # Get batch [1, D, H, W]
            _D = batch.shape[1]
            out_slices = []
            
            # Loop over slices
            for d in tqdm(range(_D), desc="Iterating slices", leave=False):

                # Get single slice - Shape [1, 1, H, W]
                slice_d = batch[:, d:d+1, :, :]

                # Histogram matching
                if num_references > 0:
                    slice_d         = slice_d.numpy()
                    slice_d_matched = flow_matching_util.match_cumulative_cdf_batch(
                        source=slice_d, template_batch=reference_trajectories_np[:, s_target, 0, 0, ...]
                    ) # (H, W) X (N, H, W) -> (H, W)
                    slice_d_matched = slice_d_matched.astype(np.float32)
                    slice_d         = torch.from_numpy(slice_d_matched[None, None]).to(device)  # To (1, 1, H, W) cuda tensor
                else:
                    slice_d = slice_d.to(device)    # To cuda tensor

                
                # Generate trajectory
                inference_trajectory = ode_solver.solve(
                    x=slice_d,
                    t_span=torch.linspace(0, 1, num_timesteps, device=device),
                )
                pred_slice = inference_trajectory[-1]
                out_slices.append(pred_slice.squeeze().cpu().numpy()) 
                
            # Stack to volume [D, H, W]
            out_vol = np.stack(out_slices)
            out_vol = out_vol.clip(-1, 1) / 2 + 0.5
            
            # Save reconstruction
            save_path = out_dir / f"pred_{idx:03d}.npy"
            np.save(save_path, out_vol)
            
    log.info(f"Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()