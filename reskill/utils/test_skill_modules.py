
import torch
import gym
import pdb
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from reskill.models.rnvp import R_NVP, stacked_NVP
from reskill.utils.general_utils import AttrDict
import reskill.rl.envs
import argparse


class TestSkillModules():
    def __init__(self, dataset_name, task):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load skill vae model
        skill_vae_path   = "results/saved_skill_models/" + dataset_name + "/skill_vae.pth"
        self.skill_vae = torch.load(skill_vae_path, map_location=torch.device(self.device))
        self.skill_vae.eval()
            
        # Load skill prior rnvp model
        skill_prior_path = "results/saved_skill_models/" + dataset_name + "/skill_prior.pth"
        self.skill_prior = torch.load(skill_prior_path, map_location=torch.device(self.device))
        self.skill_prior.eval()

        if task == "block":
            self.env = gym.make("FetchPlaceMultiGoal-v0")
        elif task == "hook":
            self.env = gym.make("FetchHook-v0")


        self.n_features = self.skill_vae.n_z
        self.seq_len = self.skill_vae.seq_len


    def get_obs(self, obs):
        # Multi-goal Fetch env
        obs = torch.FloatTensor(np.concatenate((obs["observation"], obs['desired_goal']))).unsqueeze(dim=0).to(self.device)
        # Fixed goal Fetch env
        #obs = torch.FloatTensor(obs["observation"]).unsqueeze(dim=0).to(device)
        return obs


    def test(self, use_skill_prior=True):

        obs = self.env.reset()
        obs = self.get_obs(obs)
        steps = 0
        
        # Multivariate Normal Distribution
        base_mu, base_cov = torch.zeros(self.n_features), torch.eye(self.n_features)
        base_dist = MultivariateNormal(base_mu, base_cov)

        while(True):

            if use_skill_prior:
                # Sample from NVP skill prior
                sample = AttrDict(noise=base_dist.rsample(sample_shape=(1,)), state=obs)
                z = self.skill_prior.inverse(sample).noise.detach()
            else:
                # Sample from entire skills latent space
                z = torch.normal(0, 1, size=(1, self.n_features)).to(self.device)
                
            for _ in range(self.seq_len):

                obs_z = torch.cat((obs, z), 1)
                a = self.skill_vae.decoder(obs_z).cpu().detach().numpy()[0]
                
                obs, reward, done, debug_info = self.env.step(a)
                obs = self.get_obs(obs)

                self.env.render()

                steps += 1

                if steps > self.env._max_episode_steps or done:

                    obs = self.env.reset()
                    obs = self.get_obs(obs)
                    steps = 0
                    break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    parser.add_argument('--use_skill_prior', type=bool, default=True)
    parser.add_argument('--task', type=str, default="block", choices=["block", "hook"])
    args = parser.parse_args()
    t = TestSkillModules(args.dataset_name, args.task)
    t.test(use_skill_prior=args.use_skill_prior)