
import argparse
import numpy as np
from reskill.utils.controllers.pick_and_place_controller import get_pick_and_place_control
from reskill.utils.controllers.push_controller import get_push_control
from reskill.utils.controllers.hook_controller import get_hook_control
import gym
from tqdm import tqdm
from reskill.utils.general_utils import AttrDict
import reskill.rl.envs
from tqdm import tqdm
import random
import os
from perlin_noise import PerlinNoise


class CollectDemos():
    """
    Class to generate a dataset of demonstrations for the Fetch environment tasks using a set of handcrafted controllers.
    
    """
    def __init__(self, dataset_name, num_trajectories=5, subseq_len=10, task="block"):

        self.seqs = []
        self.task = task
        self.dataset_dir = "../dataset/" + dataset_name + "/"
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.save_dir = "../dataset/" + dataset_name + "/" + "demos.npy"
        self.num_trajectories = num_trajectories
        self.subseq_len = subseq_len
        if self.task == "hook":
            self.env = gym.make('FetchHook-v0')
        else:
            self.env = gym.make('FetchPlaceMultiGoal-v0')

        

    def get_p_noise(self, idx, factor):
        a = np.array([self.x_noise(idx/factor), self.y_noise(idx/factor), self.z_noise(idx/factor), 0])
        return a

    def get_obs(self, obs):
        return np.concatenate([obs['observation'], obs['desired_goal']])


    def collect(self):
        print("Collecting demonstrations...")
        for i in tqdm(range(self.num_trajectories)):

            obs = self.env.reset()
            done = False
            actions = []
            observations = []
            terminals = []

            self.x_noise = PerlinNoise(octaves=3)
            self.y_noise = PerlinNoise(octaves=3)
            self.z_noise = PerlinNoise(octaves=3)

            if self.task == "block":
                controller = random.choice([get_pick_and_place_control, get_push_control])
            else:
                controller = get_hook_control

            idx = 0

            while not done:

                o = self.get_obs(obs)
                observations.append(o)

                p_noise = self.get_p_noise(idx, 1000)
                idx += 1

                action, success = controller(obs)

                action += p_noise * 0.5
                actions.append(action)

                obs, _, done, _ = self.env.step(action)
                terminals.append(success)

                #self.env.render()

                if success:
                    break

            if len(actions) <= self.subseq_len+1:
                continue
            else:
                self.seqs.append(AttrDict(
                    obs=observations,
                    actions=actions,
                    ))

        
        np_seq = np.array(self.seqs)
        np.save(self.save_dir, np_seq)
                
        print("Dataset Generated.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=10)
    parser.add_argument('--subseq_len', type=int, default=10)
    parser.add_argument('--task', type=str, default="block", choices=["block", "hook"])
    args = parser.parse_args()

    dataset_name = "fetch_" + args.task + "_" + str(args.num_trajectories)
    collector = CollectDemos(dataset_name=dataset_name, num_trajectories=args.num_trajectories, subseq_len=args.subseq_len, task=args.task)
    collector.collect()


