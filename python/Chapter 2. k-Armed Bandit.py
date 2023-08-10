import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 12, 4

class Bandit:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def roll(self):
        return np.random.normal(self.mean, self.std)
    
    def random_walk(self, step_size=0.1):
        if np.random.rand() > 0.5:
            mean = self.mean+step_size
        else:
            mean = self.mean-step_size
        return Bandit(mean, self.std)


def random_bandits(k, max_mean=1.0, max_std=0.1):
    bs = []
    for _ in range(k):
        mean = np.random.uniform(0.0, max_mean)
        std = np.random.uniform(0.0, max_std)
        bs.append(Bandit(mean, std))
    return bs
    
    
def generate_bandits_list(k, n):    
    """Generate list of n k-armed bandits
       Used to create non stationary process
    """
    bandits = random_bandits(k)
    xs = [bandits]
    for i in range(1, n):
        bs = [b.random_walk() for b in xs[i-1]]
        xs.append(bs)
    return xs

    
def plot_bandits(bs):
    k = len(bs)
    x = np.array(np.linspace(1, k, k))
    y = [x.mean for x in bandits]
    e = [x.std for x in bandits]
    plt.errorbar(x, y, e, linestyle='None', marker='o')
    plt.show()    
    
    
def moving_average(xs, n=100):
    cs = np.cumsum(xs, dtype=float)
    cs[n:] = cs[n:] - cs[:-n]
    return cs[n - 1:] / n    


# Sample bandits
K = 10
bandits = random_bandits(K)
bs = [b.mean for b in bandits]
print('Best bandit is %d with the mean=%f' % (np.argmax(bs)+1, np.max(bs)))
plot_bandits(bandits)

class Agent:
    
    def choose_action(self):
        return 0
    
    def learn(self, action, reward):
        pass
    
    
def evaluate_stationary(agent, bandits, num_episodes=1000, window=10):
    """ Stationary process. Always use the same bandits
        Return percentage score based on the highest mean from all bandits means.
    """
    best_score = np.max([b.mean for b in bandits])
    rewards = []
    for episode in range(num_episodes):
        total_reward = 0
        for j in range(100):
            a = agent.choose_action()
            reward = bandits[a].roll()
            agent.learn(a, reward)
            total_reward += reward
        rewards.append(total_reward/(100*best_score))
    return moving_average(rewards, window)


def evaluate(agent, bandits_list, window=10):
    """ Non stationary process. Each episode uses different bandits
        Return 'moving' percentage of maximum value. 
        This percentage depends on the selected bandits
    """
    rewards = []
    for bandits in bandits_list:
        best_score = np.max([b.mean for b in bandits])
        total_reward = 0
        for j in range(100):
            a = agent.choose_action()
            reward = bandits[a].roll()
            agent.learn(a, reward)
            total_reward += reward
        rewards.append(total_reward/(100.0*best_score))
    return moving_average(rewards, window)

class RandomAgent(Agent):
    
    def __init__(self, k):
        self.k = k
        
    def choose_action(self):
        """ Choose which bandit to run. 
        """
        return np.random.choice(range(self.k))        

agent = RandomAgent(K)    
score = 100 * np.mean([b.mean for b in bandits]) / np.max([b.mean for b in bandits])
print('Expected score for this agent: %0.2f%%' % score)
ax = plt.plot(evaluate_stationary(RandomAgent(K), bandits))
plt.ylim([0, 1])
plt.show()

class EGreedyAgent(Agent):
    
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.counters = np.zeros(k)
        
    def choose_action(self):
        """Select random bandit with probability epsilon.
           Or best bandit otherwise
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.k))
        else:
            return np.argmax(self.Q)
        
    def learn(self, action, reward):
        n = self.counters[action]
        self.Q[action] = (self.Q[action]*n + reward)/(n+1)
        self.counters[action] += 1

plt.plot(evaluate_stationary(RandomAgent(K), bandits), color='red', label='Random')
plt.plot(evaluate_stationary(EGreedyAgent(K, 0.0), bandits), color='orange', label='Epsilon=0')
plt.plot(evaluate_stationary(EGreedyAgent(K, 0.1), bandits), color='blue', label='Epsilon=0.01')
plt.plot(evaluate_stationary(EGreedyAgent(K, 0.01), bandits), color='lightgreen', label='Epsilon=0.01')
plt.legend()
plt.show()

class AlphaAgent:
    
    def __init__(self, k, epsilon, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)
        
    def choose_action(self):
        """Select random bandit with probability epsilon.
           Or best bandit otherwise
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.k))
        else:
            return np.argmax(self.Q)
        
    def learn(self, action, reward):
        self.Q[action] = self.Q[action] + self.alpha*(reward-self.Q[action])

plt.plot(evaluate_stationary(EGreedyAgent(K, 0.01), bandits), color='orange', label='E-Greedy')
plt.plot(evaluate_stationary(AlphaAgent(K, 0.01), bandits), color='blue', label='Alpha')
plt.legend()
plt.ylim([0, 1.2])
plt.show()

bandits_list = generate_bandits_list(K, 1000)
plt.plot(evaluate(AlphaAgent(K, 0.1), bandits_list), color='blue', label='Alpha agent')
plt.plot(evaluate(EGreedyAgent(K, 0.1), bandits_list), color='orange', label='E-Greedy agent')
plt.legend()
plt.show()

class UCBAgent(Agent):
    
    def __init__(self, k, c=2.0):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.counters = np.zeros(k)
        
    def choose_action(self):
        if np.min(self.counters) == 0:
            return np.argmin(self.counters)
        else:
            t = np.sum(self.counters)
            var = self.c * np.sqrt(np.log(t)/self.counters)
            return np.argmax(self.Q + var)
        
    def learn(self, action, reward):
        n = self.counters[action]
        self.Q[action] = (self.Q[action]*n + reward)/(n+1)
        self.counters[action] += 1

plt.plot(evaluate_stationary(UCBAgent(K), bandits), color='green', label='UCB')
plt.plot(evaluate_stationary(AlphaAgent(K, 0.1), bandits), color='orange', label='Alpha')
plt.legend()
plt.show()

bandits_list = generate_bandits_list(K, 1000)
plt.plot(evaluate(UCBAgent(K), bandits_list), color='green', label='UCB')
plt.plot(evaluate(AlphaAgent(K, 0.1), bandits_list), color='orange', label='Alpha')
plt.legend()
plt.show()

class GradientAgent(Agent):
    
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.H = np.zeros(k)
        self.Q = np.zeros(k)
        
    def choose_action(self):
        return np.random.choice(self.k, 1, p=self.softmax())[0]
        
    def learn(self, action, reward):
        self.Q[action] = self.Q[action] + 0.1*(reward-self.Q[action])
        pi = self.softmax()
        for i in range(self.k):
            if i == action:
                self.H[i] = self.H[i] + self.alpha*(reward-self.Q[i])*(1.0-pi[i])
            else:
                self.H[i] = self.H[i] - self.alpha*(reward-self.Q[i])*pi[i]
        
    def softmax(self):
        H_exp = np.exp(self.H)
        return H_exp / np.sum(H_exp)

plt.plot(evaluate_stationary(GradientAgent(K, alpha=0.1), bandits), color='green', label='Gradient')
plt.plot(evaluate_stationary(UCBAgent(K), bandits), color='blue', label='UCB')
plt.plot(evaluate_stationary(AlphaAgent(K, epsilon=0.1), bandits), color='orange', label='Alpha')
plt.legend()
plt.show()

bandits_list = generate_bandits_list(K, 1000)
plt.plot(evaluate(GradientAgent(K, alpha=0.1), bandits_list), color='blue', label='Gradient')
plt.plot(evaluate(UCBAgent(K), bandits_list), color='green', label='UCB')
plt.plot(evaluate(AlphaAgent(K, 0.1), bandits_list), color='orange', label='Alpha')
plt.legend()
plt.show()

