from __future__ import print_function
import gym
import numpy as np

env = gym.make('CartPole-v0')
print("Highs:", env.observation_space.high)
print("Lows: ", env.observation_space.low)

print(env.action_space)

observation = env.reset()
observation

# Here I'm just keeping track of my personal best. This has to be updated manually.
# ... When I got this, it converged after ~75 batches, w/ these params:
#   discount_factor = 0.9
#   batch_size = 100
#   learning_rate = 0.15
#   max_episode_length = 5000
# This seems to usually converge after between 70 and 200 batches.
personal_best_reward = 5000
personal_best_weight = np.array([  6.94065202,  83.09736598,  54.54100834,  68.92081203])

#Hyperparameters
discount_factor = 0.9  # Reward decay for rewards given after the action.
batch_size = 100
learning_rate = 0.1
max_episode_length = 5000

def random_range(a,b,shape):
    return (b-a) * np.random.random(shape) + a

# Create the initial weight vector, of the same shape as the input.
W = random_range(-1,1,env.observation_space.shape)

def sigmoid(x): 
    from scipy.special import expit
    return expit(x)  # sigmoid "squashing" function to interval [0,1]



def model_forward_step(W,x):
    ''' Simplest model ever: Just the linear multiplication, i.e. dot product! '''
    y_prob = sigmoid(np.dot(W,x))
    action = 1 if np.random.uniform() < y_prob else 0
    return action, y_prob
model_forward_step(W,observation)

def model_backward_step(x,y_prob,action_taken,reward):
    ''' Calculate dreward_dW:
    If reward is positive, we want to make the *action we took* *more likely*, if negative, make it less likely.
    So if reward is positive, we want to increase y_prob to be more towards action_taken, by reward amount.
    So our gradient will be how to adjust W to make y_prob more like action_taken. *reward.
    '''
    # Assume action_taken = 1, y_prob = 0.9, reward = +1
    chance = action_taken-y_prob  # 0.1
    dreward_dyprob = chance*reward # 
    
    dyprob_dW = x
    dreward_dW = dreward_dyprob*dyprob_dW
    return dreward_dW
model_backward_step(observation, 0.1, 1, 1)



def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    current_gamma = discount_factor
    reverse_discounted_sum = 0
    for t in reversed(xrange(0,len(rewards))):
        reverse_discounted_sum *= discount_factor
        reverse_discounted_sum += float(rewards[t])
        discounted_rewards[t] = reverse_discounted_sum
    return discounted_rewards
discount_rewards([2,2.,2], 0.1)

# Create the initial weight vector, of the same shape as the input.
W = random_range(-1,1,env.observation_space.shape)
total_reward = 0.0
episode_number = 0
batch_number = 0
Ws = [np.copy(W)]  # Just to keep track so you can try playing with the weights from each step.
batch_rewards = []  # To keep track for printing, plotting, etc.
running_avg_rewards = [] # To keep track for printing, plotting, etc.

# Start by printing any previous runs so you can start & stop w/out losing
# output history.
for i in range(len(batch_rewards)):
    print("{0}: Avg Batch reward: {1:.5}, running avg: {2:.6}".format(i, batch_rewards[i], running_avg_rewards[i]))
try:
    while True:
        gradient = np.zeros_like(W)
        total_batch_reward = 0.0
        for ep in range(0,batch_size):
            observation = env.reset()
            done = False
            total_episode_reward = 0
            observations = []
            rewards = []
            y_probs = []
            actions_taken = []
            #for _ in range(max_episode_length):
            while True:
                #env.render()
                action, y = model_forward_step(W,observation)
                observations.append(observation)
                y_probs.append(y)
                actions_taken.append(action)

                observation, reward, done, info = env.step(action)

                rewards.append(reward)
                total_episode_reward += reward

                if done:
                    break

            # End of the Episode
            episode_number += 1
            discounted_ep_rewards = discount_rewards(rewards, discount_factor)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_ep_rewards -= np.mean(discounted_ep_rewards)
            discounted_ep_rewards /= np.std(discounted_ep_rewards)

            ep_grad = np.zeros_like(W)
            for i in range(0, len(observations)):
                ep_grad += model_backward_step(observations[i],
                                               y_probs[i],
                                               actions_taken[i],
                                               discounted_ep_rewards[i])

            gradient += ep_grad
            total_batch_reward += total_episode_reward

        # End of batch
        total_reward += total_batch_reward
        running_avg_reward = total_reward/((batch_number+1)*batch_size)
        batch_rewards.append(total_batch_reward/batch_size)
        running_avg_rewards.append(running_avg_reward)

        if (batch_number % 1) == 0:
            print("{0}: Avg Batch reward: {1:.5}, running avg: {2:.6}".format(batch_number, batch_rewards[batch_number], running_avg_rewards[batch_number]))
 
        W += learning_rate * gradient
        gradient = np.zeros_like(W) # reset batch gradient buffer
 
        batch_number += 1
        Ws.append(np.copy(W))

except KeyboardInterrupt:
    print("Stopping Looping!")
    print("Interrupted loop ({0}): interrupted episode reward: {1:.5}, info: {2}".format(batch_number, total_episode_reward, info))

num_batches = batch_number+1
average_reward = total_reward/num_batches
print("Num Batches: {0}, Avg Reward: {1}".format(num_batches,average_reward))
print("Final Weights:", W)

Ws[-1]





def render_model(W, num_steps=max_episode_length*2, num_test_episodes = 5):
    total_reward = 0
    
    for i_episode in range(num_test_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        print("{0}/{1}:".format(i_episode, num_test_episodes))
        for _ in range(num_steps):
            env.render()  # I don't think you can get this to render from MyBinder. :(
            action,_ = model_forward_step(W,observation)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        print("{0}/{1}: Episode Reward: {2}".format(i_episode, num_test_episodes, episode_reward))
        total_reward += episode_reward
    average_reward = total_reward/num_test_episodes
    return total_reward,average_reward

render_model(random_range(-1,1,env.observation_space.shape), num_steps=10000, num_test_episodes = 100)

render_model(Ws[33], num_steps=10000, num_test_episodes = 3)

render_model(personal_best_weight, num_steps=10000, num_test_episodes = 2)











