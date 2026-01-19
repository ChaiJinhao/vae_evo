import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
import  os
from scipy.spatial.distance import cdist




# set fitness target and distance scale to unify the scale and slope of the fitness
fitness_target = {
    "rosenbrock": 0,
    "beale": 0,
    "himmelblau": 0,
    "ackley": 12.5401,
    "rastrigin": 64.6249, # x_i = 3.51786 max=0 for all dimensions
    "rastrigin_4d": 129.2498,
    "rastrigin_32d": 1033.9980,
    "rastrigin_256d": 8271.9844
}


distance_scale = {
    "rosenbrock": 287.51,
    "beale": 20,
    "himmelblau": 17.01,
    "ackley": 2,
    "rastrigin": 30,
    "rastrigin_4d": 60,
    "rastrigin_32d": 500,
    "rastrigin_256d": 4000
}


def evaluate(model,input_dim, obj, obj_name: str, k=64, maximize=False):

    with torch.no_grad():
        # Uniform sampling of the initial population
        z = torch.randn(10000, input_dim)
        x_generated, *rest = model(z)

    pop = x_generated
    fitness = obj(x_generated)

    # top-k pop
    top_fitness, indices = torch.topk(fitness, k, largest=maximize)
    top_pop = pop[indices]




    # entropy
    entropy_all = point_entropy(pop,fitness)
    entropy_top_k = point_entropy(pop,fitness,n=64)

    # fitness
    target = fitness_target[obj_name]
    scale = distance_scale[obj_name]
    map_fun = energy_wrapper(obj, target=target, scale=scale)
    fitness_reward_all = map_fun(pop).mean()
    fitness_reward_top_k = map_fun(top_pop).mean()

    #AVGF
    avgf_all = AVGF(fitness_reward_all, entropy_all)
    avgf_top_k = AVGF(fitness_reward_top_k,entropy_top_k)

    results = {
        "entropy": {
            "all": entropy_all,
            "top_k": entropy_top_k
        },
        "fitness_reward": {
            "all": fitness_reward_all,
            "top_k": fitness_reward_top_k
        },
        "avgf": {
            "all": avgf_all,
            "top_k": avgf_top_k
        }
    }


    return results





def select_from_each_population(x_generated, obj, k, maximize=False):
    """
    Select k//n of the best individuals from each subpopulation

    Args:
        x_generated: list of tensors, each shape [N_i, D]
        obj: objective function (fitness function)
        k: total number of individuals to select
        maximize: True if higher fitness is better

    Returns:
        best_pop: selected individuals, shape [k, D]
        best_fitness: their fitness values, shape [k]
    """
    n_pops = len(x_generated)
    if n_pops == 0:
        raise ValueError("x_generated is empty")


    per_pop = k // n_pops
    remainder = k % n_pops

    selected_pops = []
    selected_fitness = []

    for i, pop in enumerate(x_generated):
        fitness = obj(pop)

        num_select = per_pop + (1 if i < remainder else 0)

        num_select = min(num_select, len(pop))

        if num_select > 0:
            best_vals, indices = torch.topk(fitness, num_select, largest=maximize)
            selected_pops.append(pop[indices])
            selected_fitness.append(best_vals)


    if selected_pops:
        best_pop = torch.cat(selected_pops, dim=0)
        best_fitness = torch.cat(selected_fitness, dim=0)
    else:
        best_pop = torch.empty(0, x_generated[0].shape[-1])
        best_fitness = torch.empty(0)

    return best_pop, best_fitness










def evaluate_muti_peak(model,input_dim, obj, obj_name: str, k=64, maximize=False):

    with torch.no_grad():
        # Uniform sampling of the initial population
        z = torch.randn(10000, input_dim)
        x_generated, *rest = model(z)


    top_pop, top_fitness = select_from_each_population(x_generated, obj, k=k, maximize=maximize)

    pop = torch.cat(x_generated, dim=0)
    fitness = obj(pop)


    # entropy
    entropy_all = point_entropy(pop,fitness)
    entropy_top_k = point_entropy(top_pop,top_fitness)

    # fitness
    target = fitness_target[obj_name]
    scale = distance_scale[obj_name]
    map_fun = energy_wrapper(obj, target=target, scale=scale)
    fitness_reward_all = map_fun(pop).mean()
    fitness_reward_top_k = map_fun(top_pop).mean()

    #AVGF
    avgf_all = AVGF(fitness_reward_all, entropy_all)
    avgf_top_k = AVGF(fitness_reward_top_k,entropy_top_k)

    results = {
        "entropy": {
            "all": entropy_all,
            "top_k": entropy_top_k
        },
        "fitness_reward": {
            "all": fitness_reward_all,
            "top_k": fitness_reward_top_k
        },
        "avgf": {
            "all": avgf_all,
            "top_k": avgf_top_k
        }
    }


    return results

















## There is a constrained optimization evaluation function



palette = {
    'total_loss': '#4C78A8',
    'fitness': '#E45756',
    'constraint': '#54A24B',
    'penalty': '#F58518',
    'highlight': '#2C3E50'
}




def evaluate_constrained(x_generated, fitness_function, constraints, epsilon=1e-4):
    with torch.no_grad():
        obj = fitness_function(x_generated)
        cons = constraints(x_generated)
        cv = torch.zeros(len(x_generated), device=x_generated.device)

        if 'ineq' in cons and cons['ineq']:
            for g in cons['ineq']:
                cv += torch.relu(g)

        if 'eq' in cons and cons['eq']:
            for h in cons['eq']:
                cv += torch.abs(h)

        if 'boundary' in cons and cons['boundary']:
            for b in cons['boundary']:
                cv += b

        feasible = cv <= epsilon
        feasible_obj = obj[feasible]

        feasibility_ratio = feasible.sum().item() / len(x_generated)
        mean_cv = cv.mean().item()
        best_feasible_obj = feasible_obj.min().item() if feasible.any() else None

        return {
            "feasibility_ratio": round(feasibility_ratio, 6),
            "cv": round(mean_cv, 6),
            "best_obj": best_feasible_obj,
        }







## Multi-objective optimization evaluation function



from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator


def evaluate_multi_objective(true_pf, approximate_pf):

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    true_pf = to_numpy(true_pf)
    approx_pf = to_numpy(approximate_pf)

    # GD
    gd = GD(true_pf)
    gd_value = gd.do(approx_pf)

    # IGD
    igd = IGD(true_pf)
    igd_value = igd.do(approx_pf)

    # SP (Spacing)
    spacing = SpacingIndicator()
    sp_value = spacing.do(approx_pf)

    return {
        'GD': gd_value,
        'IGD': igd_value,
        'SP': sp_value
    }






















## entropy


def get_top_values(values, pop, n):

    assert values.ndim == 1, "The values must be one-dimensional (N, )"
    assert values.size(0) == pop.size(0), "The first dimension sizes of values and items must be the same."

    #
    _, indices = torch.topk(values, k=n, largest=True, sorted=False)

    #
    return pop[indices]


def prob(x, scale=10):
    classification = torch.round(x * scale).long()
    # count the number of points in each class, return [class, num]
    classes, num = torch.unique(classification, return_counts=True, dim=0)
    prob = num.float() / num.sum()
    return prob


def entropy(x, scale=10):
    p = prob(x, scale)
    return torch.sum(-p * torch.log2(p))


def point_entropy(x, fitness, n=None, scale=10):
    """
    Calculate the distribution entropy of the given population

    Args:
        x: torch.Tensor, shape (N, D) —— The last generation's solution
        fitness: torch.Tensor, shape (N) —— The fitness value corresponding to the last generation's solution
        n: int, optional —— top-n fitness
        scale: int —— Grid refinement degree

    Returns:
        float —— entropy
    """
    # Only retain the top-n optimal solutions
    if n is not None:
        x = get_top_values(fitness, x, n)


    return entropy(x, scale).item()



# reward


def top_reward(fitnesses, n=None):

    if n is not None:
        last_gen = fitnesses[-1]

        top_n = torch.topk(last_gen, n).values

        return top_n.mean().item()

    else:
        return fitnesses.mean().item()




# Adaptive Variance-Gated Fitness

import torch

def AVGF(fitness, entropy, mu0=0.7, alpha=2.0, gamma=1.0):
    """
    Compute AVGF score using PyTorch.

    Args:
        fitness (torch.Tensor): shape [N], fitness values
        entropy (float or torch.Tensor): entropy value (scalar)
        mu0 (float): threshold for mean fitness
        alpha (float): scaling factor for entropy term
        gamma (float): gain for sigmoid


    Returns:
        float or torch.Tensor: AVGF score
    """
    mu = torch.mean(fitness)

    if isinstance(entropy, (int, float)):
        entropy = torch.tensor(entropy)
    entropy = torch.clamp(entropy, min=1e-8)

    if mu > mu0:
        f_entropy = alpha * torch.sigmoid(gamma * entropy)
    else:
        f_entropy = alpha * torch.sigmoid(-gamma * entropy)

    result = mu * (1 + f_entropy)

    return result.item()


### [0-1] reward

def energy_wrapper(
    obj,
    target=0.0,
    scale=1.0,
    epsilon=1e-8
):
    """
    energy = 1 / (1 + |obj(x) - target| / scale)
    range: (0, 1]
    """
    def wrapped_obj(x):
        distance = torch.abs(obj(x) - target) / scale
        energy = 1.0 / (1.0 + distance + epsilon)
        return energy

    return wrapped_obj


def energy_wrapper_np(obj, target=0.0, scale=1.0, epsilon=1e-8):


    def wrapped_obj(x):

        obj_vals = obj(x)

        distance = np.abs(obj_vals - target) / scale

        energy = 1.0 / (1.0 + distance + epsilon)

        return energy

    return wrapped_obj



def set_seed(seed=42):
    """ set_seed """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ['PYTHONHASHSEED'] = str(seed)



