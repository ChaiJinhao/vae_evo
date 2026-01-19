import torch
from tqdm import tqdm
from probfun.fitness import rosenbrock as fitness_function
from probfun import mapping
import torch.optim as optim
from utils.loss import diversity_loss
from utils.muon import SingleDeviceMuonWithAuxAdam



#VAE

def train_IVAE(
    model,
    device,
    input_dim,
    batch_size,
    fitness_function = fitness_function,
    optimization = "Adam",
    mapping_type="exp",
    objective='minimize',
    alpha=1.0,  # used for rational
    vae_weight=0.25,
    div_weight=0.0, 
    global_radius=5.0,
    epochs=2000,
    T=1.0,  # used for exp, sigmoid
    log_interval=10,
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

    # Probability mapping function
    prob_map = mapping.ProbMapping(method=mapping_type)

   

    loss_history = []
  
    progress_bar = tqdm(range(epochs), desc="Training", total=epochs)

    for epoch in progress_bar:

        # Sampling of random population
        x = (torch.rand(batch_size, input_dim, device=device) - 0.5) * 2 * global_radius

        x_recon, mu, logvar = model(x)

        # Calculate fitness
        if objective == "minimize":
         fitness_x_recon = prob_map.forward(-fitness(x_recon)/T).clamp(min=1e-6)
        elif objective == "maximize":
         fitness_x_recon = prob_map.forward(fitness(x_recon)/T).clamp(min=1e-6)
        else:
            raise ValueError(f"Unsupported method: {objective}")


        loss_div = diversity_loss(x_recon)
     
        loss_fitness = -torch.log(fitness_x_recon).mean()
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = loss_fitness + vae_weight * kl_div + div_weight * loss_div


        # backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient Clipping (Preventing Gradient Explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.cpu().item())

        if (epoch + 1) % log_interval == 0 or epoch == 0:
          progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss.item():.6f}")
       
    


    print("Training completed.")
    return model, loss_history






# VQVAE




def train_VQVAE(
    model,
    device,
    input_dim,
    batch_size,
    fitness_function = fitness_function,
    optimization = "Adam",
    mapping_type="exp",
    objective='minimize',
    alpha=1.0,  # used for rational
    vae_weight=0.1,
    div_weight=0.0, 
    global_radius=5.0,
    epochs=2000,
    T=1.0,  # used for exp, sigmoid
    log_interval=10,
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
        mapping_type:  ['exp', 'sigmoid', 'rational']
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

    # Probability mapping function
    prob_map = mapping.ProbMapping(method=mapping_type, T=T)

   

    loss_history = []
  
    progress_bar = tqdm(range(epochs), desc="Training", total=epochs)

    for epoch in progress_bar:

        # Sampling of random population
        x = (torch.rand(batch_size, input_dim, device=device) - 0.5) * 2 * global_radius

        x_recon,  vq_loss = model(x)

        # Calculate fitness
        if objective == "minimize":
         fitness_x_recon = prob_map(-fitness(x_recon)/T).clamp(min=1e-6)
        elif objective == "maximize":
         fitness_x_recon = prob_map(fitness(x_recon)/T).clamp(min=1e-6)
        else:
            raise ValueError(f"Unsupported method: {objective}")
            

        loss_div = diversity_loss(x_recon)
        loss_fitness = -torch.log(fitness_x_recon).mean()
        total_loss = loss_fitness + vae_weight * vq_loss + div_weight * loss_div


        # backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient Clipping (Preventing Gradient Explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.cpu().item())

        if (epoch + 1) % log_interval == 0 or epoch == 0:
          progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss.item():.6f}")
       
    


    print("Training completed.")
    return model, loss_history