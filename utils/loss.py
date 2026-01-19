import torch

import torch.nn.functional as F





# Single-head VAE

# Loss of diversity: Encouraging a wider distribution of generated sample sets
def diversity_loss(x_t):
    # Calculate the mean of pairwise distances
    pdist = torch.cdist(x_t, x_t)
    return -pdist.mean()  # The greater the distance, the smaller the loss.




# Multi-head VAE

# Maximization of population distance
def head_diversity_loss(x_t_list):
    # x_t_list: [head_num, batch, dim]
    loss = 0
    n = len(x_t_list)
    for i in range(n):
        for j in range(i+1, n):
            loss -= torch.norm(x_t_list[i].mean(0) - x_t_list[j].mean(0))
    return loss


def head_diversity_loss_harm(x_t_list):
    means = torch.stack([x_t.mean(0) for x_t in x_t_list])
    dist_matrix = torch.cdist(means, means, p=2)
    n = dist_matrix.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=means.device)
    dists = dist_matrix[mask]
    # Harmonic mean
    harm_mean = len(dists) / (1e-6 + (1.0 / (dists + 1e-6)).sum())
    return -harm_mean

# Minimization of within-group variance
def intra_species_loss(x_t_list):
    # x_t_list: [head_num, batch, dim]
    loss = 0
    for x_t in x_t_list:

        loss += x_t.var(dim=0).sum()
    return -loss


# MUTI-OBJECT VAE




def fit_loss(f1, f2, batch_size):
    """
    Quantitative loss: The weighted sum method approximates the solution to the Pareto frontier
    """
    w = torch.rand(batch_size)  # 随机权重 [0,1]
    return (w * f1 + (1 - w) * f2).mean()


# Joint variance loss

def torch_cov(x):
    """
    x: [B, D]
    Covariance matrix [D, D]
    """
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    cov = torch.mm(x_centered.T, x_centered) / (x.size(0) - 1)
    return cov


#

def robust_diversity_score(f1, f2):
    """
    Polymerization degree: The larger the value, the more polymerized it is (worse), and the smaller the value, the more dispersed it is.
    """
    f = torch.stack([f1, f2], dim=1)
    B = len(f)
    if B < 2:
        return torch.tensor(1.0, device=f.device)

    # Calculate covariance and determinant
    cov = torch.cov(f.T)
    det_raw = torch.det(cov)

    # Computing covariance and determinant dynamic normalization: Using the range of f1 + f2
    f_sum = f1 + f2
    scale = f_sum.max() - f_sum.min() + 1e-6
    scale_sq = scale ** 2

    det_norm = det_raw / scale_sq
    aggregation = 1.0 / (det_norm + 1e-6)  #

    return aggregation

def compressed_diversity_loss(f1, f2, beta=0.1):
    raw_aggregation = robust_diversity_score(f1, f2)
    compressed = F.softplus(raw_aggregation)  #
    return beta * compressed

