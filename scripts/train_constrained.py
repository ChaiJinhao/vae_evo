import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils.muon import SingleDeviceMuonWithAuxAdam
from model.VAEs import ConstrainedVAE
from probfun import mapping
from tqdm import tqdm
from scripts.eval import evaluate_constrained












def train(
    model,
    device,
    input_dim,
    batch_size,
    fitness_function,
    constraint_fn,
    bound_min = -2.0,
    bound_max = 2.0,
    optimization = "Adam",
    beta_start = 0.1,
    beta_end = 0.5,
    global_radius=4.0,
    epochs=2000,
    log_interval=100,
    learning_rate=3e-4,
):
    """
    Training function
    
    parameter description：

        model: choose form VAEs
        device: 'cuda' or 'cpu'
        input_dim: 
        batch_size: 
        epochs: 
        global_radius: Sampling range [-global_radius, global_radius]
        T: Temperature parameters, used for fitness mapping
        fitness_function: input: x_t; return: fitness
        mapping_type:  ['exp', 'rational', 'sigmoid', 'identity']
        alpha: rational 
        vae_weight: Loss weight of Vector quantization or KL divergence 
        log_interval: How many epochs should the log be printed

    """
    # Load the model and the optimizer


    model = model.to(device)
    model.train()

    if optimization == "Adam":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimization == "Muon":
        model_weights = [p for p in model.parameters() if p.ndim >= 2]
        model_biases = [p for p in model.parameters() if p.ndim < 2]

        param_groups = [
                        dict(params=model_weights, use_muon=True,
                        lr=0.02, weight_decay=0.01),
                        dict(params=model_biases, use_muon=False,
                        lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01),
                        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unsupported optimization method: {optimization}")



    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # objective function
    fitness = fitness_function

   

    #training history
    train_history = {
                      'total_loss': [],
                      'fitness_mean': [],
                      'fitness_std': [],
                      'best_obj':[],
                      'feasibility':[],
                    }


    progress_bar = tqdm(range(epochs), desc="Training", total=epochs)

    for epoch in progress_bar:

        # Dynamically adjust the KL weights
        current_beta = beta_start + (beta_end - beta_start) * (epoch / epochs)

        # Sampling of random population
        x = (torch.rand(batch_size, input_dim, device=device) - 0.5) * 2 * global_radius

        # loss
        x_recon, mu, logvar = model(x, min_val=bound_min, max_val=bound_max)

        metrics = evaluate_constrained(x_recon, fitness_function, constraint_fn, epsilon=1e-3)
        feasibility = metrics['feasibility_ratio']
        best_obj = metrics['best_obj']



        loss_fitness = torch.mean(fitness(x_recon))

        fitness_mean = torch.mean(fitness(x_recon)).item()
        fitness_std = torch.std(fitness(x_recon), unbiased=True).item()

        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Augmented Lagrangian Loss
        total_loss = model.augmented_lagrangian(loss_fitness + current_beta * kl_div, x_recon, constraint_fn = constraint_fn)


        # backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient Clipping (Preventing Gradient Explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Update the multiplier
        constraint_violation = model.update_multipliers(x_recon,constraint_fn = constraint_fn)

        # Record training history
        train_history['total_loss'].append(total_loss.item())
        train_history['fitness_mean'].append(fitness_mean)
        train_history['fitness_std'].append(fitness_std)
        train_history['best_obj'].append(best_obj)
        train_history['feasibility'].append(feasibility)




        
        if epoch % log_interval == 0:
            print(f"Epoch {epoch}: Loss={total_loss.item():.4f}, "
                  f"Fitness={loss_fitness.item():.4f}")
       
    


    print("Training completed.")
    return model, train_history