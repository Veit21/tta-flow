###########################################################################
#
# Script for training the Flow Matching model.
#
###########################################################################

import logging
import torch
import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Adjust these imports to match your exact file structure
from util.datasets import FlowMatchingInferenceDataset, get_transforms, load_data_from_csv
from torch.utils.data import DataLoader
from util import model_util, flow_matching_util
from util.checkpoint_manager import CheckpointManager

@hydra.main(version_base=None, config_path="../preferences", config_name="config")
def main(cfg: DictConfig):

    # Setup logging
    log = logging.getLogger(__name__)
    
    exp_dir = Path(cfg.experiment_path)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist.")
        
    # Set up I/O
    out_dir = exp_dir / "inference" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the transforms
    _, _, test_transforms  = get_transforms(config=cfg)

    # Load the data
    test_dataframe     = load_data_from_csv(csv_path=cfg["data"]["data_dir"])
    test_dataset       = FlowMatchingInferenceDataset(
        dataframe=test_dataframe,
        transform=test_transforms,
    )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model
    model   = model_util.build_model(config=cfg, logger=log, device=device)
    ckpt_root = exp_dir / "checkpoints"
    checkpoint_manager  = CheckpointManager(checkpoint_dir=ckpt_root, logger=log)
    ckpt_path = Path(checkpoint_manager.find_latest_checkpoint())
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
    
    model.to(device)
    model.eval()

    # Initialize ODE solver
    ode_solver = flow_matching_util.ODESolver(
        model=deepcopy(model), 
        solver=cfg["train_parameters"]["solver"], 
        sample_x=cfg["loss"]["target"] == "x"
    )
    num_timesteps = 51


    # Inference Loop
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Processing Volumes")):
            
            # Get batch [1, D, H, W]
            vol = batch.to(device)  
    
            _D = vol.shape[1]
            out_slices = []
            
            # Loop over slices
            for d in tqdm(range(_D), desc="Iterating slices", leave=False):

                # Get single slice - Shape [1, 1, H, W]
                slice_d = vol[:, d:d+1, :, :] 
                
                # Generate trajectory
                inference_trajectory = ode_solver.solve(
                    x=slice_d,
                    t_span=torch.linspace(0, 1, num_timesteps, device=device),
                )

                # Normalize to [0, 1]
                inference_trajectory = inference_trajectory.clip(-1, 1)
                inference_trajectory = inference_trajectory / 2 + 0.5
                pred_slice = inference_trajectory[-1]
                out_slices.append(pred_slice.squeeze().cpu().numpy()) 
                
            # Stack to volume [D, H, W]
            out_vol = np.stack(out_slices)
            out_vol = out_vol / 2 + 0.5
            
            # Save reconstruction
            save_path = out_dir / f"pred_{idx:03d}.npy"
            np.save(save_path, out_vol)
            
    log.info(f"Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()