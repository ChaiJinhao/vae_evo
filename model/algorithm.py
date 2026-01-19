import torch

## GA

class GA:
    def __init__(self, fitness_function=None, pop_size=100, dim=2, generations=1000,
                 crossover_rate=0.8, mutation_rate=0.05, ub=4, lb=-4, mode='max', device="cuda"):
        self.pop_size = pop_size
        self.dim = dim
        self.fitness_function = fitness_function
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.device = device
        self.mode = mode
        self.upper_bound = ub
        self.lower_bound = lb

        # Initialize the population
        self.pop = torch.rand(pop_size, dim, device=device) * (self.upper_bound - self.lower_bound) + self.lower_bound
        self.best_fitness = {'max': -float('inf'), 'min': float('inf')}[mode]
        self.best_individual = None
        self.fitness_history = []

    def evaluate(self):
        """Calculate fitness"""
        return  self.fitness_function(self.pop)

    def selection(self, fitness):
        """
        Selection operation: Roulette selection, supporting maximization and minimization problems.

        Args:
            fitness (torch.Tensor): Fitness value
            mode (str): 'max' , 'min'

        Returns:
            torch.Tensor: The population composed of the selected individuals
        """
        if self.mode == 'max':
            # Max: The greater the fitness, the higher the probability.
            prob = fitness - fitness.min() + 1e-6
        else:  # mode == 'min'
            # Min: The smaller the fitness value, the better. → Reverse the situation, making the small ones become large.
            prob = fitness.max() - fitness + 1e-6

        # Normalize to probability
        prob = prob / prob.sum()

        # Poker wheel sampling (repetitive)
        idx = torch.multinomial(prob, self.pop_size, replacement=True)

        return self.pop[idx]




    def crossover(self, selected):
        """crossover"""
        offspring = []
        for i in range(0, self.pop_size, 2):
            p1, p2 = selected[i], selected[(i + 1) % self.pop_size]
            if torch.rand(1).item() < self.crossover_rate:
                alpha = torch.rand(self.dim, device=self.device)
                child1 = alpha * p1 + (1 - alpha) * p2
                child2 = alpha * p2 + (1 - alpha) * p1
                offspring.extend([child1, child2])
            else:
                offspring.extend([p1, p2])
        return torch.stack(offspring)



    def mutation(self, offspring):
        """mutation"""
        mutation_mask = (torch.rand(self.pop_size, self.dim, device=self.device) < self.mutation_rate)
        offspring = offspring + mutation_mask * (torch.rand(self.pop_size, self.dim, device=self.device) * 2 - 1)
        # boundary treatment
        return torch.clamp(offspring, self.lower_bound, self.upper_bound)




    def _update_best(self, fitness):
        if self.mode == 'min':
            current = fitness.min().item()
            idx = fitness.argmin()
        else:  # max
            current = fitness.max().item()
            idx = fitness.argmax()

        if (self.mode == 'min' and current < self.best_fitness) or \
                (self.mode == 'max' and current > self.best_fitness):
            self.best_fitness = current
            self.best_individual = self.pop[idx].detach().cpu().numpy()



    def run(self):

        for gen in range(self.generations):

            fitness = self.evaluate()


            self._update_best(fitness)


            self.fitness_history.append(self.best_fitness)

            selected = self.selection(fitness)
            offspring = self.crossover(selected)
            self.pop = self.mutation(offspring)



        return self.best_individual, self.best_fitness, self.pop


# # 打印进度
# if (gen + 1) % 20 == 0:
#     print(f"第{gen + 1}代，当前最优值: {self.best_fitness:.4f}, 最优个体: {self.best_individual}")


### PSO


class Particle:
    def __init__(self, position_dim, mode='min', lower_bound=-4, upper_bound=4):
        """
        mode: 'min' , 'max'
        """
        self.lb = lower_bound
        self.ub = upper_bound
        self.position = torch.FloatTensor(position_dim).uniform_(self.lb, self.ub)
        self.velocity = torch.FloatTensor(position_dim).uniform_(-1, 1)
        self.best_position = self.position.clone()

        # Initialize best_fitness according to the optimization direction

        self.best_fitness = {'max': -float('inf'), 'min': float('inf')}[mode]

        self.fitness = self.best_fitness
        self.mode = mode

    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        if global_best_position is None:

            r1 = torch.rand_like(self.position)
            cognitive_component = cognitive_weight * r1 * (self.best_position - self.position)
            self.velocity = inertia_weight * self.velocity + cognitive_component
        else:
            # update speed
            r1, r2 = torch.rand_like(self.position), torch.rand_like(self.position)
            cognitive_component = cognitive_weight * r1 * (self.best_position - self.position)
            social_component = social_weight * r2 * (global_best_position - self.position)
            self.velocity = inertia_weight * self.velocity + cognitive_component + social_component



    def update_position(self):
        self.position += self.velocity
        self.position = torch.clamp(self.position, self.lb, self.ub)

    def evaluate_fitness(self, function):
        self.fitness = function(self.position).item()

        # Update the individual optimal based on the mode.
        if (self.mode == 'min' and self.fitness < self.best_fitness) or \
                (self.mode == 'max' and self.fitness > self.best_fitness):
            self.best_fitness = self.fitness
            self.best_position = self.position.clone()








class PSO:
    def __init__(self, num_particles, position_dim, function, lower_bound = -4, upper_bound=4, mode='min'):
        """
        main PSO
        """
        self.num_particles = num_particles
        self.position_dim = position_dim
        self.function = function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mode = mode


        self.global_best_fitness = {'max': -float('inf'), 'min': float('inf')}[mode]
        self.global_best_position = None

        # Create a particle swarm
        self.particles = [Particle(position_dim, self.mode, lower_bound=self.lower_bound, upper_bound=self.upper_bound) for _ in range(num_particles)]

        # Initialize the fitness of all particles
        self._evaluate_all()
        # Update the global optimum
        self.update_global_best()

    def _evaluate_all(self):

        for particle in self.particles:
            particle.evaluate_fitness(self.function)

    def update_global_best(self):

        for particle in self.particles:
            if (self.mode == 'min' and particle.fitness < self.global_best_fitness) or \
               (self.mode == 'max' and particle.fitness > self.global_best_fitness):
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.clone()

    def optimize(self, num_iterations, inertia_weight=0.5, cognitive_weight=2.0, social_weight=2.0):

        for iter_idx in range(num_iterations):
            for particle in self.particles:
                particle.update_velocity(
                    self.global_best_position,
                    inertia_weight,
                    cognitive_weight,
                    social_weight
                )
                particle.update_position()
                particle.evaluate_fitness(self.function)

            # Update the global optimum after each iteration
            self.update_global_best()

        return self.global_best_position, self.global_best_fitness, self.particles