import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

li = []
with open(r"E:\SHIVAM MAHAJAN\Desktop\Github\Beer-Recommendation-and-Application-of-MAB\Data\convergence_softmax.pickle","rb") as handle:
    for i in range(5):
        li.append(pickle.load(handle))

best_arm = li[0][3]
arm_chosen = [ans[0] for ans in li]
prob_best_arm = [np.sum(np.array(arm == best_arm), axis = 0)/5000.0 for arm in arm_chosen]

colors = ["r","b","g","c","y"]
get_ipython().magic('matplotlib inline')
plt.figure(num=None, figsize=(8,6))
labels = ["temperature 0.1","temperature 0.2", "temperature 0.3", "temperature 0.4", "temperature 0.5"] #
for i in range(len(colors)):
    plt.plot(prob_best_arm[i], colors[i], label = labels[i])
plt.title("Convergence of Softmax algo with Probability of chosing the best arm at every trial") #
plt.xlabel("Horizons")
plt.ylabel("Probability of chosing best arm")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Covergence_with_prob_of_chosing_best_arm_softmax") #

reward = [ans[1] for ans in li]
avg_reward = [np.sum(rew, axis = 0)/5000.0 for rew in reward]

colors = ["r","b","g","c","y"]
get_ipython().magic('matplotlib inline')
plt.figure(num=None, figsize=(8,6))
labels = ["temperature 0.1","temperature 0.2", "temperature 0.3", "temperature 0.4", "temperature 0.5"]
for i in range(len(colors)):
    plt.plot(avg_reward[i], colors[i], label = labels[i])
plt.title("Convergence of Softmax algo with Reward obtained every trial")
plt.xlabel("Horizons")
plt.ylabel("Avg Reward at every trial")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Covergence_with_avg_reward_softmax")

cum_reward = [ans[2] for ans in li]
avg_cum_reward = [np.sum(cr, axis = 0)/5000.0 for cr in cum_reward]

colors = ["r","b","g","c","y"]
get_ipython().magic('matplotlib inline')
plt.figure(num=None, figsize=(8,6))
labels = ["temperature 0.1","temperature 0.2", "temperature 0.3", "temperature 0.4", "temperature 0.5"]
for i in range(len(colors)):
    plt.plot(avg_cum_reward[i], colors[i], label = labels[i])
plt.title("Convergence of Softmax algo with cumulative reward sum")
plt.xlabel("Horizons")
plt.ylabel("Cumulative Reward per trial")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("Covergence_with_avg_cum_reward_softmax")



