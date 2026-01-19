import torch
from torch.distributions import MultivariateNormal




def rosenbrock(x, a=1, b=100):
    """
        Rosenbrock 
        Parameters:
                   x (Tensor): Input tensor, the last dimension must be 2 [..., x0, x1]
                   a (float): Default  1
                   b (float): Default  100
        Return:
        Tensor: Function value
    """
    if x.shape[-1] != 2:
        raise ValueError("The last dimension must be 2.")
    x0, x1 = x[..., 0], x[..., 1]
    return (a - x0)**2 + b * (x1 - x0**2)**2

def beale(x):
    """
    Beale 
    Parameters:
               x (Tensor): Input tensor, the last dimension must be 2
    Return:
            Tensor: Function value 
    """
    if x.shape[-1] != 2:
        raise ValueError("The last dimension must be 2.")
    x0, x1 = x[..., 0], x[..., 1]
    term1 = (1.5 - x0 + x0*x1)**2
    term2 = (2.25 - x0 + x0*x1**2)**2
    term3 = (2.625 - x0 + x0*x1**3)**2
    return term1 + term2 + term3

def himmelblau(x):
    """
     Himmelblau 
     Parameters:
               x (Tensor): Input tensor, the last dimension must be 2
     Return:
               Tensor: Function value
    """
    if x.shape[-1] != 2:
        raise ValueError("The last dimension must be 2.")
    x0, x1 = x[..., 0], x[..., 1]
    return (x0**2 + x1 - 11)**2 + (x0 + x1**2 - 7)**2



def ackley(x, a=20, b=0.2, c=2*torch.pi, bounds=(-4.0, 4.0), penalty_value=1e-6):
    """
    Ackley 
    Parameters:
             x (Tensor): Input tensor of any dimension
             a : Default  20
             b : Default  0.2
             c : Default  2π
    Return:
              Tensor: Function value 
    """
    min_val, max_val = bounds

    # Check if each sample is within bounds
    within_bounds = ((x >= min_val) & (x <= max_val)).all(dim=1)

    n = x.shape[-1]
    sum_sq = torch.sum(x ** 2, dim=-1)
    cos_term = torch.sum(torch.cos(c * x), dim=-1)

    # Calculate Ackley function value
    ackley_value = -a * torch.exp(-b * torch.sqrt(sum_sq / n)) - torch.exp(cos_term / n) + a + torch.e

    # Apply boundary penalty
    result = torch.where(within_bounds, ackley_value, torch.tensor(penalty_value, device=x.device, dtype=x.dtype))

    return result


def rastrigin(x, A=10, bounds=(-4.0, 4.0), penalty_value=1e-6):
    """
    Rastrigin
    parameter:
       x (Tensor): Input tensor of any dimension
       A (float): Default 10
    Return:
       Tensor: Function value
    """
    min_val, max_val = bounds

    within_bounds = ((x >= min_val) & (x <= max_val)).all(dim=1)


    n = x.shape[-1]
    original_values = A * n + torch.sum(x ** 2 - A * torch.cos(2 * torch.pi * x), dim=1)

    result = torch.where(within_bounds, original_values, torch.tensor(penalty_value, device=x.device))

    return result




import numpy as np


def rastrigin_np(x, A=10):
    """
    Rastrigin function (NumPy version)

    Parameters:
        x (array-like): Input array of any shape, where the last dimension is treated as the feature dimension
        A (float): Parameter of the Rastrigin function, default is 10

    Returns:
        numpy.ndarray: Function values, shape is the same as input except the last dimension is reduced
    """
    x = np.asarray(x)  # Convert to numpy array
    n = x.shape[-1]  # Dimensionality along last axis
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=-1)





def two_peak_1d(x):
    return torch.exp(-(x - 3) ** 2) + torch.exp(-(x + 3) ** 2)

def one_peak_1d(x):
    return torch.exp(-(x - 3) ** 2)   


def two_peak_density(x, mu1=None, mu2=None, std=0.5):
    device = x.device
    if mu1 is None:
        mu1 = torch.tensor([0., -1.], device=device)
    if mu2 is None:
        mu2 = torch.tensor([1., 0.], device=device)

    # Checking if the input tensor x has shape (2,) and unsqueeze to make it (*N, 2)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    # Covariance matrix for the Gaussian distributions (identity matrix, since it's a standard Gaussian)
    covariance_matrix = torch.eye(2, device=device) * (std ** 2)

    # Create two multivariate normal distributions
    dist1 = MultivariateNormal(mu1.to(device), covariance_matrix)
    dist2 = MultivariateNormal(mu2.to(device), covariance_matrix)

    max_prob = (dist1.log_prob(mu1).exp() + dist2.log_prob(mu2).exp()).to(device)

    # Evaluate the density functions for each distribution and sum them up
    density = (dist1.log_prob(x).exp() + dist2.log_prob(x).exp()).to(device)

    return density / max_prob * 2