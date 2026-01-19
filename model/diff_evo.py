import torch
from tqdm import tqdm
from model.diffevo import DDIMScheduler, BayesianGenerator, DDIMSchedulerCosine, DDPMScheduler, RandomProjection, LatentBayesianGenerator



def diffevo(obj, mode='max', num_pop=256, num_step=500, scaling=4.0, temperatures=None, disable_bar=False, dim=2,
               scheduler=None):
    if scheduler is None:
        scheduler = DDIMSchedulerCosine(num_step=num_step)
    else:
        scheduler = scheduler(num_step=num_step)




    x = torch.randn(num_pop, dim)

    trace = []
    x0_trace = []
    fitnesses = []
    x0_fitness = []

    for t, alpha in tqdm(scheduler, total=num_step - 1, disable=disable_bar):



        if mode == 'min':
            fitness = torch.sigmoid(-obj(x * scaling))
        elif mode == 'max':
            fitness = obj(x * scaling)

        fitnesses.append(fitness)
        generator = BayesianGenerator(x, fitness, alpha)
        x, x0 = generator(noise=0.001, return_x0=True)



        if mode == 'min':
            x0_fit = torch.sigmoid(-obj(x0 * scaling))
        elif mode == 'max':
            x0_fit = obj(x0 * scaling)

        x0_fitness.append(x0_fit)
        trace.append(x.clone() * scaling)
        x0_trace.append(x0.clone() * scaling)

    if mode == 'min':
        fitness = torch.sigmoid(-obj(x * scaling))
    elif mode == 'max':
        fitness = obj(x * scaling)


    fitnesses.append(fitness)
    x0_fitness.append(x0_fit)

    pop = x * scaling
    trace = torch.stack(trace)
    x0_trace = torch.stack(x0_trace)
    fitnesses = torch.stack(fitnesses)
    x0_fitness = torch.stack(x0_fitness)
    return pop, trace, x0_trace, fitnesses, x0_fitness




def diffevo_latent(obj, mode = 'max', num_pop=256, num_step=100, scaling=4.0, temperatures=None, disable_bar=False, dim=2):



    scheduler = DDIMSchedulerCosine(num_step=num_step)

    x = torch.randn(num_pop, dim)

    random_map = RandomProjection(dim, 2, normalize=True)

    trace = []
    x0_trace = []
    fitnesses = []
    x0_fitness = []

    for t, alpha in tqdm(scheduler, total=num_step - 1, disable=disable_bar):



        if mode == 'min':
            fitness = torch.sigmoid(-obj(x * scaling))
        elif mode == 'max':
            fitness = obj(x * scaling)

        fitnesses.append(fitness)
        generator = LatentBayesianGenerator(x, random_map(x).detach(), fitness, alpha, density='uniform')
        x, x0 = generator(noise=0.001, return_x0=True)


        if mode == 'min':
            x0_fit = torch.sigmoid(-obj(x0 * scaling))
        elif mode == 'max':
            x0_fit = obj(x0 * scaling)

        x0_fitness.append(x0_fit)
        trace.append(x.clone() * scaling)
        x0_trace.append(x0.clone() * scaling)

    if mode == 'min':
        fitness = torch.sigmoid(-obj(x * scaling))
    elif mode == 'max':
        fitness = obj(x * scaling)


    fitnesses.append(fitness)
    x0_fitness.append(x0_fit)

    pop = x * scaling
    trace = torch.stack(trace)
    x0_trace = torch.stack(x0_trace)
    fitnesses = torch.stack(fitnesses)
    x0_fitness = torch.stack(x0_fitness)
    return pop, trace, x0_trace, fitnesses, x0_fitness
