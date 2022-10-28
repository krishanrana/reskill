
import torch
import gym
import pdb
from torchvision import transforms
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from reskill.models.rnvp import R_NVP, stacked_NVP
from reskill.utils.general_utils import AttrDict
import reskill.rl.envs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load skill vae model
dataset_name = "fetch_block_40000"
skill_vae_path   = "../results/saved_skill_models/" + dataset_name + "/skill_vae.pth"
skill_vae = torch.load(skill_vae_path, map_location=torch.device(device))
skill_vae.eval()
    
# Load skill prior rnvp model
skill_prior_path = "../results/saved_skill_models/" + dataset_name + "/skill_prior.pth"
skill_prior = torch.load(skill_prior_path, map_location=torch.device(device))
skill_prior.eval()

env = gym.make("FetchPlaceMultiGoal-v0")

n_features = skill_vae.n_z
n_substeps = skill_vae.seq_len
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)]) 

use_skill_prior = False

def get_obs(obs):
    # Multi-goal Fetch env
    obs = torch.FloatTensor(np.concatenate((obs["observation"], obs['desired_goal']))).unsqueeze(dim=0).to(device)
    # Fixed goal Fetch env
    #obs = torch.FloatTensor(obs["observation"]).unsqueeze(dim=0).to(device)
    return obs



def main():

    obs = env.reset()
    obs = get_obs(obs)
    steps = 0
    
    # Multivariate Normal Distribution
    base_mu, base_cov = torch.zeros(n_features), torch.eye(n_features)
    base_dist = MultivariateNormal(base_mu, base_cov)

    while(True):

        if use_skill_prior:
            # Sample from NVP skill prior
            sample = AttrDict(noise=base_dist.rsample(sample_shape=(1,)), state=obs)
            z = skill_prior.inverse(sample).noise.detach()
        else:
            # Sample from entire skills latent space
            z = torch.normal(0, 1, size=(1, n_features)).to(device)
            
        for _ in range(n_substeps):

            obs_z = torch.cat((obs, z), 1)
            a = skill_vae.decoder(obs_z).cpu().detach().numpy()[0]
            
            obs, reward, done, debug_info = env.step(a)
            obs = get_obs(obs)
            env.render()

            steps += 1

            if steps > env._max_episode_steps or done:

                obs = env.reset()
                obs = get_obs(obs)
                steps = 0
                break

if __name__ == "__main__":
    main()