import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

with open(r"E:\SHIVAM MAHAJAN\Desktop\Github\Beer-Recommendation-and-Application-of-MAB\Data\convergence_ucb1.pickle","rb") as handle:
    ans = pickle.load(handle)

best_arm = ans[3]
arm_chosen = ans[0]
prob_best_arm = np.sum(np.array(arm_chosen == best_arm), axis = 0)/5000.0

arm_chosen[:,249] == 0

best_arm

colors = ["r","b","g","c","y"]
get_ipython().magic('matplotlib inline')
plt.figure(num=None, figsize=(8,6))
plt.plot(prob_best_arm, colors[1], label = "UCB-1")
plt.title("Convergence of UCB-1 algo with Probability of chosing the best arm at every trial") #
plt.xlabel("Horizons")
plt.ylabel("Probability of chosing best arm")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Covergence_with_prob_of_chosing_best_arm_ucb1") #ucb1

reward = ans[1]
avg_reward = np.sum(reward, axis = 0)/5000.0

colors = ["r","b","g","c","y"]
get_ipython().magic('matplotlib inline')
plt.figure(num=None, figsize=(8,6))
plt.plot(avg_reward, colors[1], label = "UCB-1")
plt.title("Convergence of UCB-1 algo with Reward obtained every trial")
plt.xlabel("Horizons")
plt.ylabel("Avg Reward at every trial")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Covergence_with_avg_reward_ucb1")

cum_reward = ans[2]
avg_cum_reward = np.sum(cum_reward, axis = 0)/5000.0

colors = ["r","b","g","c","y"]
get_ipython().magic('matplotlib inline')
plt.figure(num=None, figsize=(8,6))
plt.plot(avg_cum_reward, colors[1], label = "UCB-1")
plt.title("Convergence of UCB-1 algo with cumulative reward sum")
plt.xlabel("Horizons")
plt.ylabel("Cumulative Reward per trial")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Covergence_with_avg_cum_reward_ucb1")



