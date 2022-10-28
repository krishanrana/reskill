
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import pdb
from reskill.utils.general_utils import AttrDict
from reskill.models.rnvp import R_NVP, stacked_NVP

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SkillVAE(nn.Module):
    """
    Variational autoencoder to embed skills into a continuous latent space. This VAE leverages a closed loop decoder
    to sequentially reconstruct each action conditioned on the corresponding state.
    
    """
    def __init__(self, n_actions=4, n_obs=43, seq_length=10, n_z=10, 
                 n_hidden=128, n_layers=1, n_propr=12, device="cuda"):
        super(SkillVAE, self).__init__()
        
        self.n_actions = n_actions
        self.n_obs = n_obs #43 #28 
        self.seq_len = seq_length
        self.n_hidden = n_hidden 
        self.n_layers = n_layers # number of LSTM layers (stacked)
        self.n_z = n_z
        self.device = device
        self.n_propr = n_propr

        self.bc_criterion = nn.MSELoss(reduction="mean")

        self.lstm = nn.LSTM(input_size = self.n_actions+self.n_obs,
                            hidden_size = self.n_hidden,
                            num_layers = self.n_layers, 
                            batch_first = True)

        self.encoder = nn.Sequential(nn.Linear(self.n_hidden, 64), nn.BatchNorm1d(64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32), nn.BatchNorm1d(32),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, self.n_z*2))
   
        self.decoder = nn.Sequential(nn.Linear(self.n_z+self.n_obs, 64), nn.BatchNorm1d(64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32), nn.BatchNorm1d(32),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, self.n_actions),
                                     nn.Tanh())
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state.to(self.device), cell_state.to(self.device))
        
    def run_inference(self,x):
        # encoding
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        x = self.hidden[0].permute(1,0,2)[:,0,:]
        out = self.encoder(x)
        return out.view(-1,2,self.n_z)

    def run_decode_batch(self,x,fn):
        batch_size, seq_len, sz = x.shape
        x = x.view(batch_size*seq_len, sz)
        actions = fn(x)
        out = actions.view(batch_size, seq_len, -1)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std)
        return sample
    
    def vae_loss(self, inputs, output, beta=0.00000001):
        bc_loss = self.bc_criterion(output.reconstruction, inputs["actions"])
        kld_loss = (-0.5 * torch.sum(1 + output.q.log_var - output.q.mu.pow(2) - output.q.log_var.exp())) * beta
        return bc_loss, kld_loss
    
    def loss(self, inputs, output):
        bc_loss, kld_loss = self.vae_loss(inputs, output)
        total_loss = bc_loss + kld_loss

        return AttrDict(bc_loss=bc_loss,
                    kld_loss=kld_loss,
                    total_loss=total_loss)

    def forward(self, x):
        states = x['obs']
        actions = x['actions']
        
        # Encoding
        x_cat = torch.cat((states, actions), 2) 
        x = self.run_inference(x_cat)
        q = AttrDict(mu=x[:,0,:],
                     log_var=x[:,1,:])

        z = self.reparameterize(q.mu, q.log_var)
        z_tiled = z.repeat(1,self.seq_len).view(actions.shape[0],self.seq_len,self.n_z)
   
        # Decoding
        # Closed loop decoding
        decode_inputs = torch.cat((states, z_tiled), 2)
        reconstruction = self.run_decode_batch(decode_inputs, self.decoder)
        
        return AttrDict(reconstruction=reconstruction, q=q, z=z)




        
        
    
