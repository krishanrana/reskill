"""
Authors: OpenAI Spinning Up

"""

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from reskill.utils.general_utils import AttrDict
import reskill.rl.agents.ppo_core as core
from reskill.rl.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from reskill.rl.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import pdb

"""
Proximal Policy Optimization (by clipping), 
with early stopping based on approximate KL
Args:
    env_fn : A function which creates a copy of the environment.
        The environment must satisfy the OpenAI Gym API.
    actor_critic: The constructor method for a PyTorch Module with a 
        ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
        module. The ``step`` method should accept a batch of observations 
        and return:
        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                        | observation.
        ``v``        (batch,)          | Numpy array of value estimates
                                        | for the provided observations.
        ``logp_a``   (batch,)          | Numpy array of log probs for the
                                        | actions in ``a``.
        ===========  ================  ======================================
        The ``act`` method behaves the same as ``step`` but only returns ``a``.
        The ``pi`` module's forward call should accept a batch of 
        observations and optionally a batch of actions, and return:
        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``pi``       N/A               | Torch Distribution object, containing
                                        | a batch of distributions describing
                                        | the policy for the provided observations.
        ``logp_a``   (batch,)          | Optional (only returned if batch of
                                        | actions is given). Tensor containing 
                                        | the log probability, according to 
                                        | the policy, of the provided actions.
                                        | If actions not given, will contain
                                        | ``None``.
        ===========  ================  ======================================
        The ``v`` module's forward call should accept a batch of observations
        and return:
        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``v``        (batch,)          | Tensor containing the value estimates
                                        | for the provided observations. (Critical: 
                                        | make sure to flatten this!)
        ===========  ================  ======================================
    ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
        you provided to PPO.
    seed (int): Seed for random number generators.
    steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
        for the agent and the environment in each epoch.
    epochs (int): Number of epochs of interaction (equivalent to
        number of policy updates) to perform.
    gamma (float): Discount factor. (Always between 0 and 1.)
    clip_ratio (float): Hyperparameter for clipping in the policy objective.
        Roughly: how far can the new policy go from the old policy while 
        still profiting (improving the objective function)? The new policy 
        can still go farther than the clip_ratio says, but it doesn't help
        on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
        denoted by :math:`\epsilon`. 
    pi_lr (float): Learning rate for policy optimizer.
    vf_lr (float): Learning rate for value function optimizer.
    train_pi_iters (int): Maximum number of gradient descent steps to take 
        on policy loss per epoch. (Early stopping may cause optimizer
        to take fewer than this.)
    train_v_iters (int): Number of gradient descent steps to take on 
        value function per epoch.
    lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
        close to 1.)
    max_ep_len (int): Maximum length of trajectory / episode / rollout.
    target_kl (float): Roughly what KL divergence we think is appropriate
        between new and old policies after an update. This will get used 
        for early stopping. (Usually small, 0.01 or 0.05.)
    logger_kwargs (dict): Keyword args for EpochLogger.
    save_freq (int): How often (in terms of gap between epochs) to save
        the current policy and value function.
"""

device = torch.device('cpu')

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        #assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in data.items()}


class PPO():
    def __init__(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, obs_dim=16, act_dim=4, act_limit=2):

        self.actor_critic = actor_critic
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamme = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.save_freq = save_freq
        self.act_limit = act_limit

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create actor-critic module
        self.ac = self.actor_critic(obs_dim, act_dim, act_limit, **ac_kwargs).to(device)

        # Sync params across processes
        sync_params(self.ac)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()


    def update(self):
        data = self.buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.' %i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        return AttrDict(LossPi=pi_l_old, 
                        LossV=v_l_old,
                        KL=kl, 
                        Entropy=ent, 
                        ClipFrac=cf,
                        DeltaLossPi=(loss_pi.item() - pi_l_old),
                        DeltaLossV=(loss_v.item() - v_l_old))


