import warnings
warnings.filterwarnings('ignore')
import gym_bandits
import gym
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

sns.set_context('talk')
sns.set_palette('colorblind')

env = gym.make("BanditTenArmedGaussian-v0")

env.r_dist

bandit_means = pd.Series([b[0] for b in env.r_dist])
bandit_means.plot(kind='bar')
plt.title('Expected Value of Each Bandit')
plt.ylabel('Q(a)')
plt.xlabel('Bandit Number')
plt.xticks(rotation=0)

bandit_means.idxmax()

iterations = 1000
epsilon = .1
policy = env.action_space.sample()
bandit_data = pd.DataFrame(index=(range(iterations)), columns=['arm', 'reward', 'epsilon',
                                                               'regret', 'policy', 'random_policy'])
bandit_data['reward'] = bandit_data['reward'].astype('float')

for i in range(iterations):
    if bool(np.random.binomial(1, epsilon)):
        arm = env.action_space.sample()
    else:
        arm = policy
    reward = env.step(arm)[1]
    regret = i*bandit_means.max() - bandit_data['reward'].ix[:i].sum()
    bandit_data.ix[i] = pd.Series(data=[arm, reward, epsilon, regret, policy, arm != policy], index=list(bandit_data.columns))
    policy = int(bandit_data.ix[:i].groupby('arm')['reward'].mean().idxmax())

bandit_data.tail()

arm_counts = bandit_data.groupby('arm')['epsilon'].count()
arm_counts.index = arm_counts.index.astype(int)
ax = arm_counts.plot(kind='bar'); plt.xticks(rotation=0); plt.title('Frequency of Selecting Bandit')

bandit_data.groupby('random_policy')['epsilon'].count().plot(kind='bar')
plt.xticks(rotation=0)

v_values = pd.DataFrame(bandit_data.groupby('arm')['reward'].mean()).join(pd.DataFrame(bandit_means), how='outer')
v_values.columns = ['q_hat', 'q_star']
v_values.plot(kind='bar')
plt.title('Estimates of Q(a) vs. Actual Q(a)')
plt.xlabel('bandit number')
plt.ylabel('reward')
plt.xticks(rotation=0)

bandit_data['regret'].plot()
plt.title('Total Regret')
plt.xlabel('iteration')

iterations = 1000
epsilon = .1
policy = env.action_space.sample()
old_regret = bandit_data['regret']
bandit_data = pd.DataFrame(index=(range(iterations)), columns=['arm', 'reward', 'epsilon', 'regret', 'policy', 'random_policy'])
bandit_data['reward'] = bandit_data['reward'].astype('float')

for i in range(iterations):
    if bool(np.random.binomial(1, epsilon)):
        arm = env.action_space.sample()
    else:
        arm = policy
    reward = env.step(arm)[1]
    regret = i*bandit_means.max() - bandit_data['reward'].ix[:i].sum()
    bandit_data.ix[i] = pd.Series(data=[arm, reward, epsilon, regret, policy, arm != policy], index=list(bandit_data.columns))
    policy = int(bandit_data.ix[:i].groupby('arm')['reward'].mean().idxmax())
    epsilon = 1./((i/50) + 10)

regret_comp = pd.DataFrame(bandit_data['regret']).join(pd.DataFrame(old_regret), lsuffix='new', rsuffix='old')
regret_comp.columns = ['e-decayed', 'e-greedy']
regret_comp.plot(); plt.title('Comparing Learning Strategies') 
plt.ylabel('Regret'); plt.xlabel('iteration')

pd.DataFrame([np.divide(regret_comp['e-decayed'],regret_comp.index),
              np.divide(regret_comp['e-greedy'],regret_comp.index)]).T.plot()
plt.title('Average Regret per Iteration')
plt.xlabel('iteration')
plt.ylabel('Average Regret')

