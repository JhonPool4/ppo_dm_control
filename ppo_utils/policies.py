# libraries: neural networks in cuda
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    @info multilayer perceptron (mlp)
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation[j] if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
  
class MLPCritic(nn.Module):
    
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        # critic neural network to estimate value function
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # in device and has grad_fn
        return torch.squeeze(self.v_net(obs), -1) # Flatten: to ensure v has right shape.       

class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # actor neural network to estimate action
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, dist, act):
        # in device and has grad_fn
        return dist.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution   

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        dist = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(dist, act)
        return dist, logp_a              


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        # build policy 
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)

        # build value function
        self.vf = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
      # compute action, vavlue funtion and  log probability
        with torch.no_grad():
            dist = self.pi._distribution(obs) # distribution
            a = dist.sample() # action
            logp_a = self.pi._log_prob_from_distribution(dist, a) # logprob
            v = self.vf(obs) # value function
        # in device and doesn´t have grad_fn
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]