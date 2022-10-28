
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn 
import torch
from reskill.utils.general_utils import AttrDict
import pdb


class R_NVP(nn.Module):
    """
    State-conditioned R-NVP skill prior model
    
    """
    def __init__(self, d, k, state_size, n_hidden, device):
        super().__init__()
        self.d, self.k = d, k
        self.device = device
        self.state_size = state_size
        self.sig_net = nn.Sequential(
                    nn.Linear(k + state_size, n_hidden),
                    nn.LeakyReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.LeakyReLU(),
                    nn.Linear(n_hidden, d - k),
                    nn.Tanh())

        self.mu_net = nn.Sequential(
                    nn.Linear(k + state_size, n_hidden),
                    nn.LeakyReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.LeakyReLU(),
                    nn.Linear(n_hidden, d - k))

        base_mu, base_cov = torch.zeros(d), torch.eye(d)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x, flip=False):
        x1, x2 = x["skill"][:, :self.k].to(self.device), x["skill"][:, self.k:].to(self.device)

        if flip:
            x2, x1 = x1, x2
        
        # forward
        sig = self.sig_net(torch.cat((x1,x["state"]), 1))
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(torch.cat((x1,x["state"]), 1))
        
        if flip:
            z2, z1 = z1, z2
        
        z_hat = torch.cat([z1, z2], dim=-1)
        
        log_pz = self.base_dist.log_prob(z_hat.cpu()).to(self.device)
        log_jacob = sig.sum(-1)
        
        return AttrDict(skill=z_hat, state=x["state"]), log_pz, log_jacob
    
    def inverse(self, Z, flip=False):
        z1, z2 = Z.noise[:, :self.k].to(self.device), Z.noise[:, self.k:].to(self.device)
        
        if flip:
            z2, z1 = z1, z2
        
        x1 = z1
        x2 = (z2 - self.mu_net(torch.cat((z1,Z.state), 1))) * torch.exp(-self.sig_net(torch.cat((z1,Z.state), 1)))
        
        if flip:
            x2, x1 = x1, x2
            
        return AttrDict(noise=torch.cat([x1, x2], -1),
                        state=Z.state)


class stacked_NVP(nn.Module):
    def __init__(self, d, k, n_hidden, state_size, n, device):
        super().__init__()
        self.bijectors = nn.ModuleList([
            R_NVP(d, k, state_size, n_hidden=n_hidden, device=device).to(device) for _ in range(n)
        ])
        self.flips = [True if i%2 else False for i in range(n)]
        
    def forward(self, x):
        log_jacobs = []
        
        for bijector, f in zip(self.bijectors, self.flips):
            x, log_pz, lj = bijector(x, flip=f)
            log_jacobs.append(lj)
        
        return x, log_pz, sum(log_jacobs)
    
    def inverse(self, z):
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=f)
        return z