###########################################################################
#
# This script contains utility classes and functions for flow matching
#
###########################################################################

# Imports
import torch

from torch import nn
from tqdm import tqdm


#----------------------------------------------------------------------------
# Helper function to enforce correct data shape

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

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
    

#----------------------------------------------------------------------------
# Custom class for numerical ODE integration

class ODESolver():
    def __init__(self, model: nn.Module, solver: str="midpoint", conditional: bool=False, sample_x: bool=False):
        """Basic ODE integrator class.
        Args:
            model (nn.Module): Neural network that parameterizes the vector field v_theta(t, x), pushing a sample x0 to x1 over time.
            solver (str, optional): ODE solver, choose from ["euler", "midpoint"]. Defaults to "midpoint".
            conditional (bool, optional): Whether to solve the ODE with conditional information.
            sample_x (bool, optional): Whether the network output is the velocity field v_hat or directly the clean sample x_hat. Defaults to False.
        """
        self.model          = model
        self.solver         = solver
        self.conditional    = conditional
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
                    x_next  = x_current + dt * v_hat                                        # 3) Integrate
                else:
                    x_next  = x_current + dt * self.model(t_current, x_current)
            
            elif self.solver == "midpoint":
                if self.sample_x:
                    x_hat_k1    = self.model(t_current, x_current)                                          # 1) Predict clean x_hat at initial point
                    t_eval      = pad_t_like_x(t_current, x_hat_k1)
                    k1          = (x_hat_k1 - x_current) / torch.clamp(1 - t_eval, min=0.05)                # 2) Reformulate to obtain velocity v_hat
                    x_mid       = x_current + 0.5 * dt * k1                                                 # 3) Get midpoint given v_hat

                    x_hat_k2    = self.model(t_current + 0.5 * dt, x_mid)                                   # 4) Predict x_hat at midpoint again
                    k2          = (x_hat_k2 - x_mid) / torch.clamp(1 - (t_eval + 0.5 * dt), min=0.05)    # 5) Get 
                    x_next      = x_current + dt * k2
                else:
                    k1      = self.model(t_current, x_current)
                    x_mid   = x_current + 0.5 * dt * k1
                    k2      = self.model(t_current + 0.5 * dt, x_mid)
                    x_next  = x_current + dt * k2
            
            x_trajectory.append(x_next)
        
        return torch.stack(x_trajectory, dim=0)
    
    def solve_conditional(self, x: torch.Tensor, y: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Numerically integrates a sample x in time w.r.t. a vector field (self.model) parameterised with a neural network.
        When the 'euler' solver is specified, the ODE dx/dt = v(x) is computed as:
        x(t+1) = x(t) + v(x(t)) * dt, where dt is specified by t_span.
        
        Args:
            x (torch.Tensor): Initial condition. Shape (B, C, H, W)
            y (torch.Tensor): Conditional information for x. Shape (B, C, H, W)
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
                x_next = x_current + dt * self.model(t_current, torch.cat((x_current, y), dim=1))
            
            elif self.solver == "midpoint":
                
                # Evaluates derivative at midpoint
                k1      = self.model(t_current, torch.cat((x_current, y), dim=1))
                x_mid   = x_current + 0.5 * dt * k1
                k2      = self.model(t_current + 0.5 * dt, torch.cat((x_mid, y), dim=1))
                x_next  = x_current + dt * k2
            
            x_trajectory.append(x_next)
        
        return torch.stack(x_trajectory, dim=0)
    
    def __call__(self, x: torch.Tensor, t_span: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calls the respective method to integrate x, either guided by y or not.

        Args:
            x (torch.Tensor): Initial condition. Shape (B, C, H, W)
            t_span (torch.Tensor): 1D vector that gives the time points at which to integrate.
            y (torch.Tensor): Conditional information for x. Shape (B, C, H, W)

        Raises:
            ValueError: Raised if self.conditional=True but no conditional data is provided.

        Returns:
            torch.Tensor: Trajectory of the integration. Shape (T, B, C, H, W)
        """
        if self.conditional:
            if not y.numel() > 0:
                raise ValueError("Conditional solver requires y to be provided.")
            return self.solve_conditional(x=x, y=y, t_span=t_span)
        else:
            return self.solve(x=x, t_span=t_span)