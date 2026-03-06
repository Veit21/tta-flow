"""
Checkpoint management utilities for training resumption and model persistence.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class CheckpointManager:
    """
    Manages checkpoint loading and saving for model training.
    
    Handles checkpoint persistence with support for resuming training from the latest checkpoint.
    Manages both primary and EMA models, along with optimizer and scheduler states.
    
    Attributes:
        checkpoint_dir (str): Directory where checkpoints are stored.
        logger (logging.Logger): Logger for informational and warning messages.
    """
    
    def __init__(self, log_dir: str, logger: logging.Logger):
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir (str): Directory path where checkpoints will be saved and loaded.
            logger (logging.Logger): Logger instance for reporting status and warnings.
        """
        self.log_dir = log_dir
        self.logger = logger
        self.checkpoint_dir = self.log_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint file in the checkpoint directory.
        
        Searches for files matching the pattern "*_step_*.pt" and returns the one with
        the highest step number.
        
        Returns:
            Optional[str]: Full path to the latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        
        if not checkpoints:
            return None
        
        # Extract step numbers and find the checkpoint with the highest step
        try:
            steps = [int(f.split('_step_')[-1].replace('.pt', '')) for f in checkpoints]
            latest_idx = steps.index(max(steps))
            return os.path.join(self.checkpoint_dir, checkpoints[latest_idx])
        except (ValueError, IndexError):
            self.logger.warning("Could not parse checkpoint filenames to find the latest checkpoint.")
            return None
    
    def load_checkpoint(
        self,
        net_model: torch.nn.Module,
        ema_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> int:
        """
        Load checkpoint and restore model, optimizer, and scheduler states.
        
        Loads the latest checkpoint from the checkpoint directory and restores all model
        weights, optimizer state, and scheduler state. Handles missing or unexpected keys
        gracefully with warnings.
        
        Args:
            net_model (torch.nn.Module): Primary model to restore weights into.
            ema_model (torch.nn.Module): EMA model to restore weights into.
            optimizer (torch.optim.Optimizer): Optimizer to restore state into.
            scheduler (torch.optim.lr_scheduler.LambdaLR): Scheduler to restore state into.
        
        Returns:
            int: Step number from the loaded checkpoint. Returns 0 if no checkpoint is found.
        
        Example:
            >>> manager = CheckpointManager("path/to/checkpoints", logger)
            >>> start_step = manager.load_checkpoint(net_model, ema_model, optim, sched)
        """
        checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None:
            self.logger.warning("No checkpoint found in checkpoint directory. Starting from scratch.")
            return 0
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return 0
        
        # Load net_model weights
        net_missing, net_unexpected = net_model.load_state_dict(checkpoint['net_model'], strict=False)
        if net_missing:
            self.logger.warning(f"Missing keys in net_model: {net_missing}")
        if net_unexpected:
            self.logger.warning(f"Unexpected keys in net_model: {net_unexpected}")
        
        # Load ema_model weights
        ema_missing, ema_unexpected = ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
        if ema_missing:
            self.logger.warning(f"Missing keys in ema_model: {ema_missing}")
        if ema_unexpected:
            self.logger.warning(f"Unexpected keys in ema_model: {ema_unexpected}")
        
        # Load optimizer and scheduler states
        try:
            optimizer.load_state_dict(checkpoint['optim'])
            scheduler.load_state_dict(checkpoint['sched'])
        except KeyError as e:
            self.logger.warning(f"Could not load optimizer/scheduler state: {e}")
        
        start_step = checkpoint.get('step', 0) + 1
        self.logger.info(f"Successfully resumed training from step {start_step}")
        
        return start_step
    
    def save_checkpoint(
        self,
        net_model: torch.nn.Module,
        ema_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        step: int,
        model_name: str = "model",
    ) -> None:
        """
        Save checkpoint with model, optimizer, and scheduler states.
        
        Args:
            net_model (torch.nn.Module): Primary model to save.
            ema_model (torch.nn.Module): EMA model to save.
            optimizer (torch.optim.Optimizer): Optimizer state to save.
            scheduler (torch.optim.lr_scheduler.LambdaLR): Scheduler state to save.
            step (int): Current training step.
            model_name (str, optional): Name identifier for the model in the checkpoint filename.
                Defaults to "model".
        
        Example:
            >>> manager.save_checkpoint(net_model, ema_model, optim, sched, step=1000, model_name="gaussian")
        """
        checkpoint = {
            'net_model': net_model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optim': optimizer.state_dict(),
            'sched': scheduler.state_dict(),
            'step': step,
        }
        
        filename = os.path.join(self.checkpoint_dir, f"{model_name}_weights_step_{step}.pt")
        torch.save(checkpoint, filename)
        self.logger.info(f"Checkpoint saved to {filename}")
