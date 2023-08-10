import torch
from torch.autograd import Variable
from torch import nn
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import matplotlib

import gym
import numpy as np
import math

seed = 100
torch.manual_seed(seed)

def estimate_advantages(rewards, values, gamma, tau):
    tensor_type = type(rewards)
    returns = tensor_type(rewards.size(0), 1)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return
        deltas[i] = rewards[i] + gamma * prev_value - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage

        prev_return = returns[i, 0]
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
    advantages = (advantages - advantages.mean()) / advantages.std()
    
    return advantages, returns

def compute_discount_returns(rewards, discount):
    """given a list of returns and discount factor(gamma), compute the discount
    returns.

    Args:
        returns(tensor of rewards):
        discount(float):

    Returns:
        returns(list of floats)

    """
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for i in reversed(range(rewards.size(0)-1)):
        returns[i] = returns[i+1]*discount + rewards[i]

    return returns

# policy, value func implementation
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, action_scale, hidden_act=F.tanh):
        """
        
        Args:
            input_dim:
            hidden_size:
            output_dim:
            action_scale (float):
            
        """
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, output_dim)
        self.std_head = nn.Linear(hidden_size, output_dim)
        self.hidden_act = hidden_act
        self.action_scale = action_scale
        
    def forward(self, x, volatile):
        x = Variable(x, volatile=volatile)
        x = self.hidden_act(self.w1(x))
        x = self.hidden_act(self.w2(x))
        mean = F.tanh(self.mean_head(x))*self.action_scale
        std = F.softplus(self.std_head(x))
        return mean, std
    
    def sample_action(self, states, volatile=True):
        mean, std = self.forward(states, volatile)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.data
    
    def evaluate_action(self, states, actions, volatile=False):
        mean, std = self.forward(states, volatile)
        dist = Normal(mean, std)
        logprob = dist.log_prob(Variable(actions))
        return logprob
    
    
class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, hidden_act = F.relu):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, output_dim)
        self.hidden_act = hidden_act
        
    def forward(self, x, volatile):
        x = Variable(x, volatile=volatile)
        x = self.hidden_act(self.w1(x))
        x = self.hidden_act(self.w2(x))
        return self.w3(x)
        
    def predict(self, states, volatile=True):
        return self.forward(states, volatile=volatile)      
        
        
#--------------------------------------------------
env = gym.make('Pendulum-v0')
env._max_episode_steps = 1000
env.seed(seed)
# experimental stuff
episodes = 10000
show = 50
gamma = 0.95
tau = 0.95
steps = 200
render = False
entropy_coefficient = 0.01

R_list = []

R_avg = -1000
# logging
state_log, action_log, reward_log, real_reward_log= [], [], [], []

# ppo parameters
ppo_epoch = 20
ppo_epsilon = 0.2
ppo_grad_norm = 0.5
# baseline parameters
baseline_epoch = 5
baseline_grad_norm = 0.5

# create policy and value function
policy = GaussianPolicy(3, 128, 1, 2.0)
p_optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-4)

baseline = ValueFunction(3,128,1)
b_optimizer = torch.optim.Adam(baseline.parameters(), lr=4e-4)

for eps in range(episodes):
    state = env.reset()
    
    del state_log[:]
    del action_log[:]
    del reward_log[:]
    del real_reward_log[:]
    
    for step in range(steps):
        if render:
            env.render()
            
        action = policy.sample_action(torch.FloatTensor(state).view(1,-1))
        next_state, reward, done, _ = env.step(np.asarray([action[0,0]]))
        
        state_log.append(torch.FloatTensor(state).view(1,-1))
        action_log.append(action)
        #reward_log.append(torch.FloatTensor([(reward+8)/8]))
        reward_log.append(torch.FloatTensor([reward]))
        real_reward_log.append(reward)
                
        state = next_state
        
        if done:
            break

    # update phase
    States = torch.cat(state_log)
    Actions = torch.cat(action_log)
    Rewards = torch.cat(reward_log)
    Rewards = (Rewards - Rewards.mean())/(Rewards.std()+1e-5)
    Returns = compute_discount_returns(Rewards, gamma)
    RealRewards = np.asarray(real_reward_log)
    
    # 1) PPO Update
    # first step we update our policy via PPO
    # before iterating through our update loop, let's compute the initial action logprob and advantage, note this
    # old_log_prob is held constant thorugh our PPO update steps, so we treat it as a constant,
    # i.e some_Variable_as_a_constant = Variable(some_Variable.data)
    old_log_prob = Variable(policy.evaluate_action(States, Actions, volatile=True).data)
    # compute advantage
    Advantages = Variable(Returns).view(-1,1) - Variable(baseline.predict(States, volatile=True).data).view(-1,1)
    # To do maybe normalize?
    Advantages = (Advantages - Advantages.mean())/(Advantages.std()+1e-5)
    for p in range(ppo_epoch):
        # compute current logprob, this forward pass holds the parameters that we want to updata
        logprob = policy.evaluate_action(States, Actions, volatile=False)
        # compute action ratio, where ratio = new_log_prob / old_log_prob
        ratio = torch.exp(logprob - old_log_prob)
        # compute surrogate 1, which is surr = ratio * advantage
        
        surr1 = ratio*Advantages
        # compute surrogate 2, which is the clipped value * advantage
        surr2 = torch.clamp(ratio, 1.0-ppo_epsilon, 1.0+ppo_epsilon)*Advantages
        # we want to maximize the clipped min between the two surrogates
        policy_loss = -torch.min(surr1, surr2).mean()
        # update
        p_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm(policy.parameters(), ppo_grad_norm)
        p_optimizer.step()
        
    
    # 2). Critic update
    # for critic, we simply regress/fit our critic predictions to the actual return
    for b in range(baseline_epoch):
        # forwrad pass to get value prediction, volatile to false as we will be updating parameters in this forward
        # pass 
        value_pred = baseline.predict(States, volatile=False)
        # compute MSE between actual returns and value prediction
        value_loss = (0.5*(Variable(Returns).view(-1,1) - value_pred.view(-1,1)).pow(2)).mean()
        # update
        b_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm(baseline.parameters(), baseline_grad_norm)
        b_optimizer.step()
        
    # computing rewards
    R_avg = 0.95*R_avg + 0.05*RealRewards.sum()
    
    R_list.append(R_avg)
    
    if R_avg >= -300.0:
        print('reached good score after {} episodes, breaking!'.format(eps))
        break
        
    
    if eps % show == 0:
        render = True
        print('episode {}, reward_sum {}, reward_avg {}'.format(eps, RealRewards.sum(), R_avg))
    else:
        render = False
        
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(R_list)

# policy, value func implementation
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, action_scale, hidden_act=F.tanh):
        """
        
        Args:
            input_dim:
            hidden_size:
            output_dim:
            action_scale (float):
            
        """
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, output_dim)
        self.std_head = nn.Linear(hidden_size, output_dim)
        self.hidden_act = hidden_act
        self.action_scale = action_scale
        
    def forward(self, x, volatile):
        x = Variable(x, volatile=volatile)
        x = self.hidden_act(self.w1(x))
        x = self.hidden_act(self.w2(x))
        mean = F.tanh(self.mean_head(x))*self.action_scale
        std = F.softplus(self.std_head(x))
        return mean, std
    
    def sample_action(self, states, volatile=True):
        mean, std = self.forward(states, volatile)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.data
    
    def evaluate_action(self, states, actions, volatile=False):
        mean, std = self.forward(states, volatile)
        dist = Normal(mean, std)
        logprob = dist.log_prob(Variable(actions))
        return logprob
    
    
class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, hidden_act = F.relu):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, output_dim)
        self.hidden_act = hidden_act
        
    def forward(self, x, volatile):
        x = Variable(x, volatile=volatile)
        x = self.hidden_act(self.w1(x))
        x = self.hidden_act(self.w2(x))
        return self.w3(x)
        
    def predict(self, states, volatile=True):
        return self.forward(states, volatile=volatile)      
        
        
#--------------------------------------------------
env = gym.make('Pendulum-v0')
env._max_episode_steps = 1000
env.seed(seed)
# experimental stuff
episodes = 10000
show = 50
gamma = 0.95
tau = 0.95
steps = 200
render = False
entropy_coefficient = 0.01

R_list = []

R_avg = -1000
# logging
state_log, action_log, reward_log, real_reward_log= [], [], [], []

# ppo parameters
ppo_epoch = 20
ppo_epsilon = 0.2
ppo_grad_norm = 0.5
# baseline parameters
baseline_epoch = 5
baseline_grad_norm = 0.5

# create policy and value function
policy = GaussianPolicy(3, 128, 1, 2.0)
p_optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-4)

baseline = ValueFunction(3,128,1)
b_optimizer = torch.optim.Adam(baseline.parameters(), lr=4e-4)

for eps in range(episodes):
    state = env.reset()
    
    del state_log[:]
    del action_log[:]
    del reward_log[:]
    del real_reward_log[:]
    
    for step in range(steps):
        if render:
            env.render()
            
        action = policy.sample_action(torch.FloatTensor(state).view(1,-1))
        next_state, reward, done, _ = env.step(np.asarray([action[0,0]]))
        
        state_log.append(torch.FloatTensor(state).view(1,-1))
        action_log.append(action)
        #reward_log.append(torch.FloatTensor([(reward+8)/8]))
        reward_log.append(torch.FloatTensor([reward]))
        real_reward_log.append(reward)
                
        state = next_state
        
        if done:
            break

    # update phase
    States = torch.cat(state_log)
    Actions = torch.cat(action_log)
    Rewards = torch.cat(reward_log)
    Rewards = (Rewards - Rewards.mean())/(Rewards.std()+1e-5)
    Returns = compute_discount_returns(Rewards, gamma)
    RealRewards = np.asarray(real_reward_log)
    
    # 1) PPO Update
    # first step we update our policy via PPO
    # before iterating through our update loop, let's compute the initial action logprob and advantage, note this
    # old_log_prob is held constant thorugh our PPO update steps, so we treat it as a constant,
    # i.e some_Variable_as_a_constant = Variable(some_Variable.data)
    old_log_prob = Variable(policy.evaluate_action(States, Actions, volatile=True).data)
    # compute advantage
    # first we need value prediction
    value_pred_old = baseline.predict(States, volatile=True).data
    Advantages, _= estimate_advantages(Rewards.view(-1,1),value_pred_old.view(-1,1),gamma, tau)
    Advantages = Variable(Advantages)
    for p in range(ppo_epoch):
        # compute current logprob, this forward pass holds the parameters that we want to updata
        logprob = policy.evaluate_action(States, Actions, volatile=False)
        # compute action ratio, where ratio = new_log_prob / old_log_prob
        ratio = torch.exp(logprob - old_log_prob)
        # compute surrogate 1, which is surr = ratio * advantage
        surr1 = ratio*Advantages
        # compute surrogate 2, which is the clipped value * advantage
        surr2 = torch.clamp(ratio, 1.0-ppo_epsilon, 1.0+ppo_epsilon)*Advantages
        # we want to maximize the clipped min between the two surrogates
        policy_loss = -torch.min(surr1, surr2).mean()
        # update
        p_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm(policy.parameters(), ppo_grad_norm)
        p_optimizer.step()
        
    
    # 2). Critic update
    # for critic, we simply regress/fit our critic predictions to the actual return
    for b in range(baseline_epoch):
        # forwrad pass to get value prediction, volatile to false as we will be updating parameters in this forward
        # pass 
        value_pred = baseline.predict(States, volatile=False)
        # compute MSE between actual returns and value prediction
        value_loss = (0.5*(Variable(Returns).view(-1,1) - value_pred.view(-1,1)).pow(2)).mean()
        # update
        b_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm(baseline.parameters(), baseline_grad_norm)
        b_optimizer.step()
        
    # computing rewards
    R_avg = 0.95*R_avg + 0.05*RealRewards.sum()
    
    R_list.append(R_avg)
    
    if R_avg >= -300.0:
        print('reached good score after {} episodes, breaking!'.format(eps))
        break
        
    
    if eps % show == 0:
        render = True
        print('episode {}, reward_sum {}, reward_avg {}'.format(eps, RealRewards.sum(), R_avg))
    else:
        render = False
        
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(R_list)

Rewards = torch.ones(10,1)
value_pred = torch.rand(10,1)
estimate_advantages(Rewards,value_pred, 0.99,0.95 )



