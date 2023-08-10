import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

rewards_mu = [
    0.3, 0.9, 0.85, 0.5
]
num_arms = len(rewards_mu)

def get_reward(i):
    return int(np.random.rand() < rewards_mu[i])

def estimate_expectation(policy, num_trials, num_reps=1000):
    def run_experiment():
        # init
        chosen_arms = np.zeros((num_trials, num_arms))
        rewards_per_arm = np.zeros((num_trials, num_arms))
        for i, choice in enumerate(np.random.permutation(num_arms).astype(int)):
            chosen_arms[i, choice] = 1
            rewards_per_arm[i, choice] = get_reward(choice)

        # play machines
        for i in range(num_arms, num_trials):        
            choice = policy(rewards_per_arm[:i, :].sum(axis=0), chosen_arms.sum(axis=0))
            chosen_arms[i, choice] = 1
            rewards_per_arm[i, choice] = get_reward(choice)
            
        return rewards_per_arm.sum(axis=1).cumsum(axis=0), chosen_arms.cumsum(axis=0)

    # repeat experiment and compute average
    avg_cum_rewards = np.zeros(num_trails)
    avg_cum_chosen_arms = np.zeros((num_trials, num_arms))
    for i in range(num_reps):
        cum_rewards, cum_chosen_arms = run_experiment()
        avg_cum_rewards += cum_rewards
        avg_cum_chosen_arms += cum_chosen_arms

    return avg_cum_rewards/float(num_reps), avg_cum_chosen_arms/float(num_reps)

def epsilon_greedy(epsilon):
    def run(total_rewards_per_arm, num_chosen_arm):
        if np.random.rand() > epsilon:
            means_per_arm = total_rewards_per_arm/num_chosen_arm
            return means_per_arm.argmax()
        else:
            return np.random.randint(0, num_arms)
    return run

def ucb1():
    def run(total_rewards_per_arm, num_chosen_arm):
        means_per_arm = total_rewards_per_arm/num_chosen_arm
        uncertainty_per_arm = np.array([np.sqrt(2*np.log(num_chosen_arm.sum()) / arm_chosen) for arm_chosen in num_chosen_arm])
        return (means_per_arm + uncertainty_per_arm).argmax()
    return run

def thompson():
    def run(total_rewards_per_arm, num_chosen_arm):
        theta = np.array([np.random.beta(total_rewards + 1, chosen_arm - total_rewards + 1) for total_rewards, chosen_arm in zip(total_rewards_per_arm, num_chosen_arm)])
        return theta.argmax()
    return run

num_trails = 1000
results = {
    'ucb1': estimate_expectation(ucb1(), num_trails),
    'thompson': estimate_expectation(thompson(), num_trails),
    'eps-greedy(0.05)': estimate_expectation(epsilon_greedy(0.05), num_trails),
    'eps-greedy(0.1)': estimate_expectation(epsilon_greedy(0.1), num_trails),
    'eps-greedy(0.5)': estimate_expectation(epsilon_greedy(0.5), num_trails),
}

draws = np.arange(num_trails) + 1
plt.figure(figsize=(14, 7))
for meth, result in results.items():
    total_reward, _ = result
    plt.semilogx(draws, max(rewards_mu)-total_reward/draws)

plt.ylabel('Regrets per Method')
plt.ylabel('normalized regret')
plt.xlabel('number of draws')
plt.legend(results.keys())

f, ax = plt.subplots(len(results), figsize=(14, len(results)*7))

for i, (meth, result) in enumerate(results.items()):
    _, chosen_arms = result
    for j in range(num_arms):
        ax[i].semilogx(chosen_arms[:, j]/draws)
    ax[i].legend(rewards_mu)
    ax[i].set_xlabel('number of draws')
    ax[i].set_ylabel('percentage of choosing an arm')
    ax[i].set_title(meth)



