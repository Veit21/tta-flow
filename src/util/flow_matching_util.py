###########################################################################
#
# Utility classes and functions for flow matching
#
###########################################################################

# Imports
import torch

import numpy as np

from torch import nn
from tqdm import tqdm


#----------------------------------------------------------------------------
# Helper functions

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.
    NOTE: Taken from torchcfm library

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

# Histogram Matching
def match_cumulative_cdf_batch(source, template_batch):
    """Return modified source array so that the cumulative density function of
    its values matches the average cumulative density function across a batch
    of template images.

    Adapted from skimage.exposure.match_histograms()
    For more information see https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html

    Args:
        source (np.ndarray): Source image to be matched
        template_batch (np.ndarray): Batch of N template images

    Returns:
        np.ndarray: Processed source image (H, W)
    """
    source          = source.squeeze()            # To shape (H, W)
    template_batch  = template_batch.squeeze()    # To shape (N, H, W)

    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        
        # Compute histogram for each template image in the batch
        max_val = template_batch.max()
        tmpl_counts_batch = np.array([
            np.bincount(template_batch[i].reshape(-1), minlength=max_val+1)
            for i in range(template_batch.shape[0])
        ])
        # Average the histograms across the batch
        tmpl_counts = tmpl_counts_batch.mean(axis=0)
        
        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = np.unique(
            source.reshape(-1), return_inverse=True, return_counts=True
        )
        
        # Collect all unique values and their counts across all template images
        all_tmpl_values = []
        all_tmpl_counts = []
        for i in range(template_batch.shape[0]):
            vals, counts = np.unique(template_batch[i].reshape(-1), return_counts=True)
            all_tmpl_values.append(vals)
            all_tmpl_counts.append(counts)
        
        # Create a unified histogram across all unique values
        all_unique_vals = np.unique(np.concatenate(all_tmpl_values))
        tmpl_counts_aggregate = np.zeros(len(all_unique_vals))
        
        for vals, counts in zip(all_tmpl_values, all_tmpl_counts):
            indices = np.searchsorted(all_unique_vals, vals)
            tmpl_counts_aggregate[indices] += counts
        
        tmpl_values = all_unique_vals
        tmpl_counts = tmpl_counts_aggregate / template_batch.shape[0]
    
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / (template_batch.shape[1] * template_batch.shape[2])
    
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(source.shape)
    

#----------------------------------------------------------------------------
# Custom class for numerical ODE integration

class ODESolver():
    def __init__(self, model: nn.Module, solver: str="midpoint", sample_x: bool=False):
        """Basic ODE integrator class.
        Args:
            model (nn.Module): Neural network that parameterizes the vector field v_theta(t, x), pushing a sample x0 to x1 over time.
            solver (str, optional): ODE solver, choose from ["euler", "midpoint"]. Defaults to "midpoint".
            sample_x (bool, optional): Whether the network output is the velocity field v_hat or directly the clean sample x_hat. Defaults to False.
        """
        self.model          = model
        self.solver         = solver
        self.sample_x       = sample_x
        if self.solver not in ["euler", "midpoint"]:
            raise NotImplementedError(f"Solver '{self.solver}' not implemented")
    
    def solve(self, x: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Numerically integrates a sample x in time w.r.t. a vector field (self.model) parameterised with a neural network.
        When the 'euler' solver is specified, the ODE dx/dt = v(x) is computed as:
        x(t+1) = x(t) + v(x(t)) * dt, where dt is specified by t_span.
        
        Args:
            x (torch.Tensor): Initial condition. Shape (B, C, H, W)
            t_span (torch.Tensor): 1D vector that gives the time points at which to integrate.
        
        Returns:
            torch.Tensor: Trajectory of the integration. Shape (T, B, C, H, W)
        """
        x_trajectory = [x]
        
        for i in tqdm(range(1, len(t_span)), desc="Integrating", disable=True):
            dt          = t_span[i] - t_span[i-1]
            t_current   = t_span[i-1]
            x_current   = x_trajectory[-1]
            
            if self.solver == "euler":
                if self.sample_x:
                    x_hat   = self.model(t_current, x_current)                              # 1) Predict clean x_hat
                    v_hat   = (x_hat - x_current) / torch.clamp(1 - t_current, min=0.05)    # 2) Reformulate to obtain v_hat
                    x_next  = x_current + dt * v_hat                                        # 3) Integration step
                else:
                    x_next  = x_current + dt * self.model(t_current, x_current)
            
            elif self.solver == "midpoint":
                if self.sample_x:
                    x_hat_k1    = self.model(t_current, x_current)                                          # 1) Predict clean x_hat at initial point
                    t_eval      = pad_t_like_x(t_current, x_hat_k1)
                    k1          = (x_hat_k1 - x_current) / torch.clamp(1 - t_eval, min=0.05)                # 2) Reformulate to obtain velocity v_hat
                    x_mid       = x_current + 0.5 * dt * k1                                                 # 3) Get midpoint given v_hat

                    x_hat_k2    = self.model(t_current + 0.5 * dt, x_mid)                                   # 4) Predict x_hat at midpoint again
                    k2          = (x_hat_k2 - x_mid) / torch.clamp(1 - (t_eval + 0.5 * dt), min=0.05)       # 5) Reformulate to v_hat again
                    x_next      = x_current + dt * k2                                                       # 6) Integration step
                else:
                    k1      = self.model(t_current, x_current)
                    x_mid   = x_current + 0.5 * dt * k1
                    k2      = self.model(t_current + 0.5 * dt, x_mid)
                    x_next  = x_current + dt * k2
            
            x_trajectory.append(x_next)
        
        return torch.stack(x_trajectory, dim=0)
    
    
    def __call__(self, x: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Calls the 'solve' method to integrate x over t_span.

        Args:
            x (torch.Tensor): Initial condition. Shape (B, C, H, W)
            t_span (torch.Tensor): 1D vector that gives the time points at which to integrate.

        Returns:
            torch.Tensor: Trajectory of the integration. Shape (T, B, C, H, W)
        """
        return self.solve(x=x, t_span=t_span)