from ppo_utils import PPOBuffer, MLPActorCritic
from torch.optim import Adam
import torch.nn as nn
import torch

from copy import copy
import pandas as pd
import numpy as np
import os

class PPO():

    def __init__(self, env, **hyperparameters):

        # update hyperparameters
        self.set_hyperparameters(hyperparameters)
        
        # get information from environment
        self.env=env
        self.obs_dim=self.env.observation_dim
        self.act_dim=self.env.action_dim

        # create neural network model
        self.ac_model=MLPActorCritic(self.obs_dim, self.act_dim, self.hidden, self.activation)

        # optimizer for policy and value function
        self.pi_optimizer=Adam(self.ac_model.pi.parameters(), self.pi_lr)
        self.vf_optimizer=Adam(self.ac_model.vf.parameters(), self.vf_lr)

        # buffer of training data
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.gamma, self.lam)

        # logger to print/save data
        self.logger={'mean_rew':0, 'std_rew':0}
      
        # create directory to save data and model
        if not os.path.exists(self.training_path):
            os.makedirs(self.training_path)
            print(f"new directory created: {self.training_path}")
        # save training data
        self.column_names=['mean', 'std']
        self.df = pd.DataFrame(columns=self.column_names,dtype=object)
        if self.create_new_training_data:
            self.df.to_csv(os.path.join(self.training_path,self.data_filename), mode='w' ,index=False)  
            print(f"new data file created: {self.data_filename}")            
        # load model
        if self.load_model:
            self.ac_model.load_state_dict(torch.load(os.path.join(self.training_path, self.model_filename)))
            print(f"model loaded: {self.model_filename}")

    def compute_loss_pi(self, data):
        # get specific training data
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # policy loss
        act_dist, logp = self.ac_model.pi(obs, act) # eval new policy
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        ent = act_dist.entropy().mean().item()
        loss_pi = -(torch.min(ratio * adv, clip_adv) + self.coef_ent*ent).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    def compute_loss_vf(self, data):
        # get specific training data
        obs, ret = data['obs'], data['ret']
        # value function loss
        return ((self.ac_model.vf(obs) - ret)**2).mean()    

    def update(self):
        # get all training data
        data = self.buf.get()

        # logger reward information
        self.logger['mean_rew']=data['mean_rews'].mean().item()
        self.logger['std_rew']=data['mean_rews'].std().item() 

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print(f"Early stooping at step {i} due to max kl")
                break
            loss_pi.backward() # compute grads
            self.pi_optimizer.step() # update parameters
    
        # Value function learning
        for i in range(self.train_vf_iters):
            self.vf_optimizer.zero_grad()
            loss_vf = self.compute_loss_vf(data)
            loss_vf.backward() # compute grads 
            self.vf_optimizer.step() # update parameters

    def rollout(self):
        # reset environemnt parameters
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        
        # generate training data
        for t in range(self.steps_per_epoch):
            # get action, value function and logprob
            a, v, logp = self.ac_model.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            self.buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = copy(next_o) # should be copy

            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t==self.steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = self.ac_model.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                self.buf.finish_path(v)

                # reset environemnt parameters
                o, ep_ret, ep_len = self.env.reset(), 0, 0
              

    def learn(self):         

        for epoch in range(self.epochs):
            # generate data
            self.rollout()

            # call update
            self.update()

            print("====================")
            print(f"epochs: {epoch+1}")        
            print(f"mean_ret: {self.logger['mean_rew']}")
            print(f"std_ret: {self.logger['std_rew']}")
            print("====================\n")

            # save reward info
            row = np.expand_dims(np.array([self.logger['mean_rew'], self.logger['std_rew']]), axis = 1).tolist()
            df_row = pd.DataFrame.from_dict(dict(zip(self.column_names, row)))
            self.df.append(df_row, sort=False).to_csv(os.path.join(self.training_path,self.data_filename), index=False, mode = 'a', header=False)  
            # save model
            if ((epoch+1)%self.save_freq==0):
                torch.save(self.ac_model.state_dict(), os.path.join(self.training_path,self.model_filename) )
                print("saving model")

            # reset logger
            self.logger={'rew_mean':0, 'rew_std':0}

    def set_hyperparameters(self, hyperparameters):
        self.epochs=1000
        self.steps_per_epoch=2000
        self.max_ep_len=400
        self.gamma=0.99
        self.lam=0.97
        self.clip_ratio=0.2
        self.target_kl=0.01
        self.coef_ent = 0.001

        self.train_pi_iters=50
        self.train_vf_iters=50
        self.pi_lr=3e-4
        self.vf_lr=1e-3

        self.hidden=(64,64)
        self.activation=[nn.Tanh, nn.ReLU]

        self.flag_render=False

        self.save_freq=500
        
        self.training_path='./training/hopper/standUp'
        self.data_filename='data'
        self.model_filename='ppo_ac_model.pth'
        self.create_new_training_data=False
        self.load_model=False        

        # change default hyperparameters
        for param, val in hyperparameters.items():
            exec("self."+param+"="+"val")  

