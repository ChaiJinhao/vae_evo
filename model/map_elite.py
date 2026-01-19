
import numpy as np

import torch




def MapElite(obj, pop_dim=2, lower_bound=-4, upper_bound=4, init_num_pop=100, num_iter=256, sigma_mut=0.1, sigma_init=1, grid_size=10):
    # https://arxiv.org/pdf/1504.04909
    assert num_iter > init_num_pop
    populations = []
    maps = dict()

    def feature_descriptor(x):
        cls = tuple(torch.round(x * grid_size).long().tolist())
        return cls

    # generate initial population
    pop_init = torch.randn(init_num_pop, pop_dim) * sigma_init
    rewards = obj(pop_init)
    for p, r in zip(pop_init, rewards):
        cls = feature_descriptor(p)
        if cls not in maps:
            maps[cls] = (p, r)
            populations.append(p)
        elif r > maps[cls][1]:
            maps[cls] = (p, r)
            populations.append(p)
    # iterate
    for i in range(num_iter - init_num_pop):
        # random select a population to mutate
        idx = np.random.randint(0, len(maps))
        p_old = list(maps.values())[idx][0]
        p_new = p_old + torch.randn(pop_dim) * sigma_mut
        p_new = torch.clamp(p_new, lower_bound, upper_bound)
        r_new = obj(p_new.unsqueeze(0)).squeeze(0)
        cls = feature_descriptor(p_new)
        if cls not in maps:
            maps[cls] = (p_new, r_new)
            populations.append(p_new)
        elif r_new > maps[cls][1]:
            maps[cls] = (p_new, r_new)
            populations.append(p_new)

    populations = torch.stack(populations)
    fitnesses = torch.stack([r for p, r in maps.values()])
    return populations, maps, fitnesses