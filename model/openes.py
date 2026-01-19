import numpy as np
import torch



class OpenES_model:
    def __init__(self, num_params, popsize, lower_bound=-5, upper_bound=5, sigma_init=1, learning_rate=1e-3, learning_rate_decay=1, sigma_decay=1,
                 momentum=0.9):
        self.num_params = num_params
        self.popsize = popsize
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.sigma = np.ones(num_params) * sigma_init
        self.learning_rate = learning_rate
        self.sigma_decay = sigma_decay
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum

        self.theta = np.zeros(num_params)
        self.velocity = np.zeros(num_params)
        self.eps = None

    def ask(self):
        self.eps = np.random.randn(self.popsize, self.num_params)
        pop = self.theta + self.sigma * self.eps
        pop = self.smooth_bound(pop)
        return pop

    def smooth_bound(self, z):
        lb = self.lower_bound
        ub = self.upper_bound
        center = (ub + lb) / 2.0
        half_range = (ub - lb) / 2.0
        return center + half_range * np.tanh(z)



    def tell(self, fitnesses):
        fitnesses = np.array(fitnesses).reshape(-1, 1)
        dmu = (fitnesses * self.eps).mean(axis=0) / self.sigma  # * (self.popsize ** 0.5)

        # Apply momentum
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * dmu
        self.theta += self.learning_rate * self.velocity

        self.sigma = self.sigma * self.sigma_decay
        self.learning_rate = self.learning_rate * self.learning_rate_decay


def OpenES(obj, pop_dim, popsize=512, num_steps=100, sigma_init=1, learning_rate=1000, lower_bound=-4, upper_bound=4):
    es = OpenES_model(
        num_params=pop_dim,
        popsize=popsize,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        sigma_init=sigma_init,
        learning_rate=learning_rate,
        learning_rate_decay=0.00001 ** (1 / num_steps),
        sigma_decay=0.01 ** (1 / num_steps),
    )

    populations = []
    fitnesses = []
    mu = []

    for i in range(num_steps):
        pop = es.ask()
        populations.append(pop)

        fitness = obj(pop)
        fitnesses.append(fitness)
        mu.append(es.theta.copy())
        es.tell(fitness)

    populations = torch.from_numpy(np.stack(populations)).float()
    fitnesses = torch.from_numpy(np.stack(fitnesses)).float()
    mu = torch.from_numpy(np.stack(mu)).float()
    return es, populations, fitnesses, mu





