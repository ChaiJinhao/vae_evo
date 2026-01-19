import torch
from tqdm import tqdm
from probfun.mutifitness import zdt1 as fitness_function
from probfun import mapping
import torch.optim as optim
from utils.muon import SingleDeviceMuonWithAuxAdam
from utils.loss import fit_loss, compressed_diversity_loss


def train(
    model,
    device,
    input_dim,
    latent_dim,
    batch_size,
    fitness_function = fitness_function,
    scale = True,
    optimization = "Adam",
    div_weight=0.005, 
    epochs=2000,
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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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




    # objective function
    fitness = fitness_function


    loss_history = []
  
    progress_bar = tqdm(range(epochs), desc="Training", total=epochs)

    for epoch in progress_bar:

        optimizer.zero_grad()

        # Sampling of random population
        z = torch.randn(batch_size, latent_dim, device=device)


        x_recon = model(z)

        if scale:
            x_scaled = (x_recon + 1) / 2
        else:
            x_scaled = x_recon


        # Calculate fitness
        f1, f2 = fitness(x_scaled)

        # fitness loss
        fitness_loss = fit_loss(f1, f2, batch_size)


        # diversity loss
        div_loss = compressed_diversity_loss(f1, f2)

        

        # Loss Function (Weighted Fitness + KL Divergence + Diversity)

        loss =  fitness_loss  + div_weight * div_loss

        if torch.isnan(loss) or torch.isinf(loss):
              continue


        # backpropagation
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        loss_history.append(loss.cpu().item())

        if (epoch + 1) % log_interval == 0 or epoch == 0:
          progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}| Scalar: {fitness_loss.item():.4f}| Div: {div_loss.item():.4f}")
       
    


    print("Training completed.")
    return model, loss_history


