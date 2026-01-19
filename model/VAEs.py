import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F





class ScaledTanh(nn.Module):
    def __init__(self, scale=4.0, alpha=2.0):
        """
        :param scale: Output range scaling factor (default 4.0)
        :param alpha: Enter the scaling factor to control the steepness of the nonlinear region (default 2.0)
        """
        super().__init__()
        self.scale = scale
        self.alpha = alpha

        
    def forward(self, x):
        return self.scale * torch.tanh(x / self.alpha)



# VAE 
class IVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=2, use_tanh=False, tanh_scale=4.0, tanh_alpha=2.0):
        super(IVAE, self).__init__()
        self.latent_dim = latent_dim
        self.use_tanh = use_tanh


        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

        # Optional ScaledTanh head
        if self.use_tanh:
            self.output_activation = ScaledTanh(scale=tanh_scale, alpha=tanh_alpha)
        else:
            self.output_activation = nn.Identity()



    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        x_recon = self.output_activation(x_recon)

        return x_recon, mu, logvar  #



# VQ-VAE
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=8, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1, 1)

    def forward(self, z):
   
        flat_z = z.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.embeddings.weight.t()))
        
       
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(z.shape)
        
       
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        
        quantized = z + (quantized - z).detach()
        return quantized, loss


class VQVAE(nn.Module):
    def __init__(self, input_dim=1, num_embeddings=128, embedding_dim=16, use_tanh=False, tanh_scale=4.0, tanh_alpha=2.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_tanh = use_tanh

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )
        
        # vq
        self.vq = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim)
        )

        if self.use_tanh:
            self.output_activation = ScaledTanh(scale=tanh_scale, alpha=tanh_alpha)
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        x_t = self.decoder(quantized)
        x_t= self.output_activation(x_t)
        return x_t, vq_loss



# MUTI-Head_VQ-VAE
class MHVQVAE(nn.Module):
    def __init__(self, input_dim=2, num_embeddings=64, embedding_dim=16, n_heads=2, use_tanh=False, tanh_scale=4.0, tanh_alpha=2.0):
        super().__init__()
        self.n_heads = n_heads

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        decoder_layers = []
        for _ in range(n_heads):
            layers = [
                nn.Linear(embedding_dim, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            ]
            if use_tanh:
                layers.append(ScaledTanh(scale=tanh_scale, alpha= tanh_alpha))
            decoder_layers.append(nn.Sequential(*layers))

        self.decoders = nn.ModuleList(decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        # 
        x_t_list = [decoder(quantized) for decoder in self.decoders]
        return x_t_list, vq_loss





# Constrained VAE model

class ConstrainedVAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=5):
        super(ConstrainedVAE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 2 * latent_dim)  
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim)
        )
        
        # lagrangian multiplier
        self.λ = nn.Parameter(torch.ones(1))  # Inequality constraint multiplier
        self.μ = nn.Parameter(torch.zeros(1))  # Equation constraint multiplier
        
        # penalty coefficient
        self.ρ = 2.0
        self.ρ_growth = 1.5
        self.ρ_max = 100.0

        self._initialize_weights()

    def _initialize_weights(self):
        """Use small-variance normal initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, min_val=-2.0, max_val=2.0):
        # encoder
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        # decoder
        x_recon = self.decoder(z)
        # Boundary constraint handling
        x_recon = torch.clamp(x_recon, min=min_val, max=max_val)
        return x_recon, mu, logvar
    

    def augmented_lagrangian(self, fitness_loss, x_recon, constraint_fn):
        """
        🔥 
        Calculate the enhanced Lagrangian loss
        param constraint_fn: Constraint function, with the format like:

        def constraint_fn(x):
            g(x) <= 0
            h(x) = 0
            -2 <= x <= 2


        return: loss
        """
        constrs = constraint_fn(x_recon)

        # inequality constraints g(x) <= 0
        ineq_viol = 0.0
        if 'ineq' in constrs and constrs['ineq']:
          for g in constrs['ineq']:
            viol = F.relu(g)
            ineq_viol += torch.mean(self.λ * viol) + 0.5 * self.ρ * torch.mean(viol**2)

        # Equation constraint h(x) = 0
        eq_viol = 0.0
        if 'eq' in constrs and constrs['eq']:
          for h in constrs['eq']:
            eq_viol += torch.mean(self.μ * h) + self.ρ * 10 * torch.mean(h**2)


        

        # boundary constraint
        bound_viol = torch.mean(constrs['boundary'][0]**2)

        return fitness_loss + ineq_viol + eq_viol + 0.1 * bound_viol


    
    def update_multipliers(self, x_tensor, constraint_fn):
        """Update Lagrange multipliers and penalty coefficients"""
        with torch.no_grad():
            constrs = constraint_fn(x_tensor)
            total_violation = 0.0
            
            # update inequality constraint multiplier
            if 'ineq' in constrs and constrs['ineq']:
              for g in constrs['ineq']:
                 viol = F.relu(g)
                 self.λ += self.ρ * torch.mean(viol)
                 total_violation += torch.mean(viol).item()
            
            # ensure λ is non-negative
            self.λ.data = torch.clamp(self.λ, min=0)
            
            # update equation constraint multiplier
            if 'eq' in constrs and constrs['eq']:
              for h in constrs['eq']:
                 self.μ += self.ρ * torch.mean(h)
                 total_violation += torch.mean(torch.abs(h)).item()
            
            # increase penalty coefficient when constraint violation is large
            if total_violation > 0.5 and self.ρ < self.ρ_max:
                self.ρ *= self.ρ_growth
                print(f"Increasing penalty coefficient ρ to {self.ρ:.2f}")
                
            return total_violation





# Multi-Objective Encoder(MOE)



class TanhToUnit(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return (torch.tanh(x) + 1) * 0.5



class MOE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128, activation=None):
        super().__init__()
        self.activation = activation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        out = self.net(z)
        if self.activation == 'tanh':
            out = torch.tanh(out)
            # If the multiple targets are LIS, then remove the comments below.
            #out = (out + 1.0) * torch.pi
        return out





  