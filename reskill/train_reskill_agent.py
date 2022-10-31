
import gym
import torch
import pdb
from tqdm import tqdm
import wandb
import time
import numpy as np
import os
from reskill.rl.utils.mpi_tools import num_procs, mpi_fork, proc_id
from reskill.rl.agents.ppo import PPO
from reskill.utils.general_utils import AttrDict
import reskill.rl.envs
import math


device = torch.device('cpu')


def get_obs(obs, env_name):
    if env_name == "FetchPyramidStack-v0": 
        out = torch.FloatTensor(np.concatenate((obs["observation"][:-6], obs["desired_goal"][-3:]))).unsqueeze(dim=0).to(device)
    else:
        out = torch.FloatTensor(np.concatenate((obs["observation"], obs["desired_goal"]))).unsqueeze(dim=0).to(device)
    return out

def logistic_fn(step, k=0.001, C=18000):
    return 1/(1 + math.exp(-k * (step-C)))

def train(agent, residual_agent, env, skill_vae, skill_prior, logistic_C, logistic_k, 
          save_path, save_path_residual):

    env_name = env.spec.id
    obs, ep_ret, ep_len = env.reset(), 0, 0
    o = get_obs(obs, env_name)

    env_step_cnt = 0
    residual_factor = 0.0

    local_steps_per_epoch = int(agent.steps_per_epoch / num_procs())

    for epoch in tqdm(range(agent.epochs)):
        for t in range(local_steps_per_epoch):
            # Select noise vector using high-level policy
            n, v, logp, mu, std = agent.ac.step(torch.as_tensor(o, dtype=torch.float32))
            sample = AttrDict(noise=n, state=o)
            # Warp noise vector to latent space skill
            z = skill_prior.inverse(sample).noise.detach()

            if proc_id() == 0:
                wandb.log({"n": n[0][0]}, env_step_cnt)
                wandb.log({"z": z[0][0]}, env_step_cnt)
                wandb.log({"mu": mu[0][0]}, env_step_cnt)
                wandb.log({"std": std[0][1]}, env_step_cnt)

            o2, skill_r = o, 0
        
            for _ in range(skill_vae.seq_len):

                obs_z = torch.cat((o2,z), 1)
                a_dec = skill_vae.decoder(obs_z)

                o_res = torch.cat((o2,z,a_dec), 1)
                a_res, v_res, logp_res, _, _ = residual_agent.ac.step(o_res)
                
                a = (a_dec.cpu().detach().numpy() + (a_res.cpu().detach().numpy() * residual_factor))[0]
                
                # Step the env
                obs, r, d, _ = env.step(a)

                env_step_cnt += 1

                if proc_id() == 0:
                    wandb.log({"action x_vel": a[0]}, env_step_cnt)
                    wandb.log({"action y_vel": a[1]}, env_step_cnt)
                    wandb.log({"residual_action x_vel": a_res[0][0]}, env_step_cnt)
                    wandb.log({"residual_action y_vel": a_res[0][1]}, env_step_cnt)
                             
                skill_r += r #Sum rewards for high level policy
                ep_ret += r
                ep_len += 1

                o2 = get_obs(obs, env_name)
                a_dec = skill_vae.decoder(torch.cat((o2,z), 1))
                o2_res = torch.cat((o2,z,a_dec), 1)

                residual_agent.buf.store(o_res.cpu().detach(), a_res.cpu().detach(), r, v_res, logp_res)

            # Update residual action weighting factor
            residual_factor = logistic_fn(env_step_cnt, k=logistic_k, C=logistic_C)
            if proc_id() == 0:
                wandb.log({"logistic_fn": residual_factor}, env_step_cnt)

            # save and log
            agent.buf.store(o.cpu().detach(), n.cpu().detach(), skill_r, v, logp)

            o = o2
            t += 1

            timeout = ep_len >= agent.max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch-1
        
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _ = agent.ac.step(o)
                    _, v_res, _, _, _ = residual_agent.ac.step(o2_res)
                else:
                    v = 0
                    v_res = 0
                agent.buf.finish_path(v)
                residual_agent.buf.finish_path(v_res)
                if terminal:
                    if proc_id() == 0:
                        wandb.log({"Episode Return": ep_ret}, env_step_cnt)
                obs, ep_ret, ep_len = env.reset(), 0, 0
                o = get_obs(obs, env_name)

        # Save model
        if (epoch % agent.save_freq == 0) or (epoch == agent.epochs-1):
            torch.save(agent.ac.pi, save_path)
            torch.save(residual_agent.ac.pi, save_path_residual)

        # Perform PPO update!
        losses = agent.update()
        residual_losses = residual_agent.update()

        if proc_id() == 0:
            wandb.log({"pi_loss_":losses.LossPi,
                        "v_loss": losses.LossV,
                        "kl": losses.KL,
                        "entropy": losses.Entropy,
                        "clip_frac":losses.ClipFrac,
                        "delta_loss_pi":losses.DeltaLossPi,
                        "delta_loss_v":losses.DeltaLossV}, env_step_cnt)



def main():
    import argparse
    import yaml
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="table_cleanup/config.yaml")
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    args=parser.parse_args()

    config_path = "configs/rl/" + args.config_file
    with open(config_path, 'r') as file:
        conf = yaml.safe_load(file)
        conf = AttrDict(conf)
    for key in conf:
        conf[key] = AttrDict(conf[key])

    mpi_fork(conf.setup.cpu)  #  run parallel code with mpi

    if proc_id() == 0:
        wandb.init(project=conf.setup.exp_name)
        wandb.run.name = conf.setup.env + "_reskill_seed_" + str(conf.setup.seed) + '_' + time.asctime().replace(' ', '_')

    env = gym.make(conf.setup.env)

    save_dir = "./results/saved_rl_models/" + args.dataset_name + "/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + "ppo_agent.pth"
    save_path_residual = save_dir + "ppo_residual_agent.pth"

    torch.set_num_threads(torch.get_num_threads())

    # Load skills module
    skill_vae_path = "./results/saved_skill_models/" + args.dataset_name + "/skill_vae.pth"
    skill_vae = torch.load(skill_vae_path, map_location=device)
    # Load skill prior module
    skill_prior_path = "./results/saved_skill_models/" + args.dataset_name + "/skill_prior.pth"
    skill_prior = torch.load(skill_prior_path, map_location=device)
    for i in skill_prior.bijectors:
        i.device = device

    n_features = skill_vae.n_z
    n_obs = skill_vae.n_obs
    seq_len = skill_vae.seq_len

    skill_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.skill_agent.hid]*conf.skill_agent.l),
                gamma=conf.skill_agent.gamma, 
                seed=conf.setup.seed, 
                steps_per_epoch=conf.skill_agent.steps_per_epoch, 
                epochs=conf.setup.epochs,
                clip_ratio=conf.skill_agent.clip_ratio, 
                pi_lr=conf.skill_agent.pi_lr,
                vf_lr=conf.skill_agent.vf_lr, 
                train_pi_iters=conf.skill_agent.train_pi_iters, 
                train_v_iters=conf.skill_agent.train_v_iters, 
                lam=conf.skill_agent.lam, 
                max_ep_len=conf.setup.max_ep_len,
                target_kl=conf.skill_agent.target_kl, 
                obs_dim=n_obs, 
                act_dim=n_features, 
                act_limit=2)
    
    residual_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.residual_agent.hid]*conf.residual_agent.l),
                        gamma=conf.residual_agent.gamma, 
                        seed=conf.setup.seed, 
                        steps_per_epoch=(conf.skill_agent.steps_per_epoch*seq_len), 
                        epochs=conf.setup.epochs,
                        clip_ratio=conf.residual_agent.clip_ratio, 
                        pi_lr=conf.residual_agent.pi_lr,
                        vf_lr=conf.residual_agent.vf_lr, 
                        train_pi_iters=conf.residual_agent.train_pi_iters, 
                        train_v_iters=conf.residual_agent.train_v_iters,
                        lam=conf.residual_agent.lam, 
                        target_kl=conf.residual_agent.target_kl, 
                        obs_dim=n_obs + n_features + env.action_space.shape[0], 
                        act_dim=env.action_space.shape[0], 
                        act_limit=1)

    print("Training RL agent...")
    train(agent=skill_agent,
          residual_agent=residual_agent,  
          env=env,
          skill_vae=skill_vae,
          skill_prior=skill_prior,
          logistic_C=conf.setup.logistic_C,
          logistic_k=conf.setup.logistic_k,
          save_path=save_path,
          save_path_residual=save_path_residual)


if __name__ == '__main__':
    main()
    
