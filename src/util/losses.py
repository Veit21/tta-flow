###########################################################################
#
# Implementations of different loss functions
#
###########################################################################

# Imports
import torch

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torch import nn
from logging import Logger


#----------------------------------------------------------------------------
# Helper functions

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.
    NOTE: Taken from torchcfm library.

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
# Standard Regression loss from Flow Matching
# See Lipman et al., Tong et al.

class FlowMatchingRegressionLoss:
    def __init__(self, logger: Logger, regression_target: str="v"):
        """Implementation of the standard regression loss of Flow Matching, i.e. E[(vt-ut)²].

        Args:
            logger (Logger): Logger for logging.
            regression_target (str): Defines the regression target. Can be either of ["x", "eps", "v"].
            Either the network learns to directly predict the clean target x, the noise eps or the velocity fied v. Defaults to "v".
        """
        self.regression_target = regression_target
        assert self.regression_target in ["x", "eps", "v"], f"Regression target not in [x, eps, v]. Got >> {self.regression_target} <<"
        if logger:
            self.logger = logger
            logger.info("Using the standard Flow Matching regression loss.")
            logger.info(f"Using regression target >> {self.regression_target} <<")

    def __call__(self, flow_matcher: ConditionalFlowMatcher, net: nn.Module, x0: torch.Tensor, x1: torch.Tensor):
        """Calculate the (batch of) interpolant(s) xt at random time points in [0, 1], get predictions from the
        network at given time points and regress the vector field.
        Using the standard regression loss from the (OT) Flow Matching framework of Lipman et al., Tong et al.

        Args:
            flow_matcher (ConditionalFlowMatcher): Specific flow matching plan to calculate the interpolant xt and the ground 
                truth vector field at time points t. The object is an instance of a class from the library 'torchcfm'.
            net (nn.Module): Neural network that predicts either the vector field, the noise, or the clean sample at position(s) xt and time(s) t.
            x0 (torch.Tensor): Batch of source data points.
            x1 (torch.Tensor): Batch of target data points.

        Returns:
            torch.Tensor: The computed regression loss for that batch.
        """

        # Sample time, interpolant and ground truth velocity at that specific time point given a specific interpolation plan 'flow_matcher'
        t, xt, ut   = flow_matcher.sample_location_and_conditional_flow(x0, x1)

        # Predict the velocity v at condition (xt, t)
        if self.regression_target == "v":
            v_hat    = net(t, xt)

        # Predict the noise eps at condition (xt, t)
        elif self.regression_target == "eps":
            eps_hat = net(t, xt)
            t_eval  = pad_t_like_x(t, xt)
            v_hat   = (xt - eps_hat) / t_eval

        # Predict the clean image x at condition (xt, t)
        elif self.regression_target == "x":
            x_hat   = net(t, xt)
            t_eval  = pad_t_like_x(t, xt)
            v_hat   = (x_hat - xt) / torch.clamp(1 - t_eval, min=0.05)
        
        else:
            raise ValueError(f"Option to regress against >> {self.regression_target} << not implemented.")

        return torch.mean((v_hat - ut) ** 2)


#----------------------------------------------------------------------------
# Other..
