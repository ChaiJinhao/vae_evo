import torch
from tqdm import tqdm
from probfun.fitness import rosenbrock as fitness_function
from probfun.mapping import ProbMapping
import torch.optim as optim
from utils.loss import head_diversity_loss_harm, intra_species_loss
from utils.muon import SingleDeviceMuonWithAuxAdam

def train(
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
    div_head_weight=0.45,
    dive_intra_weight=0.05,
    global_radius=5.0,
    epochs=2000,
    T=1.0,  # used for exp, sigmoid
    log_interval=10,
    learning_rate=3e-4,
):
    """
    The function used for training the multi-head VQ-VAE
    
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


    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # objective function
    fitness = fitness_function

    # Probability mapping function
    prob_map = ProbMapping(method=mapping_type, T=T, a=alpha)

   

    loss_history = []
  
    progress_bar = tqdm(range(epochs), desc="Training", total=epochs)

    for epoch in progress_bar:

        # Sampling of random population
        x = (torch.rand(batch_size, input_dim, device=device) - 0.5) * 2 * global_radius

        x_t_list, loss_vae = model(x)


        # Calculate fitness

        loss_fitness = 0

        if objective == "minimize":

            if mapping_type == 'exp': 

                for x_t in x_t_list:
                    fitness_xt = prob_map(fitness(x_t)/T)
                    loss_fitness += -torch.log(fitness_xt.mean() + 1e-9)


            elif mapping_type == 'rational':

                for x_t in x_t_list:
                    fitness_xt = prob_map(fitness(x_t)/T)
                    loss_fitness += -torch.log(fitness_xt.mean() + 1e-9)

            else: 

                
                for x_t in x_t_list:
                    fitness_xt = prob_map(-fitness(x_t)/T)
                    loss_fitness += -torch.log(fitness_xt.mean() + 1e-9)



        elif objective == "maximize":
                for x_t in x_t_list:
                    fitness_xt = prob_map(fitness(x_t)/T)
                    loss_fitness += -torch.log(fitness_xt.mean() + 1e-9)

        else:
            raise ValueError(f"Unsupported method: {objective}")


        loss_div_head = head_diversity_loss_harm(x_t_list)

        loss_div_inter = intra_species_loss(x_t_list)
        
  
        


        total_loss = loss_fitness + vae_weight * loss_vae + div_head_weight * torch.exp(loss_div_head) +  dive_intra_weight * torch.exp(loss_div_inter) 

        # backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient Clipping (Preventing Gradient Explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        #scheduler.step()

        loss_history.append(total_loss.cpu().item())

        if (epoch + 1) % log_interval == 0 or epoch == 0:
          progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss.item():.6f}")
       
    


    print("Training completed.")
    return model, loss_history