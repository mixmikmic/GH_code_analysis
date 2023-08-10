import numpy as np
import tensorflow as tf
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import gym
env = gym.make("CartPole-v0")

learning_rate = 1e-2 # Learning rate, applicable to both nn, policy and model

gamma = 0.99 # Discount factor for rewards

decay_rate = 0.99 # Decay factor for RMSProp leaky sum of grad**2

model_batch_size = 3 # Batch size used for training model nn
policy_batch_size = 3 # Batch size used for training policy nn

dimen = 4 # Number of dimensions in the environment

def discount(r, gamma=0.99, standardize=False):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    """
    discounted = np.array([val * (gamma ** i) for i, val in enumerate(r)])
    if standardize:
        discounted -= np.mean(discounted)
        discounted /= np.std(discounted)
    return discounted

def step_model(sess, xs, action):
    """ Uses our trained nn model to produce a new state given a previous state and action """
    # Last state
    x = xs[-1].reshape(1,-1)
    
    # Append action
    x = np.hstack([x, [[action]]])
    
    # Predict output
    output_y = sess.run(predicted_state_m, feed_dict={input_x_m: x})
    
    # predicted_state_m == [state_0, state_1, state_2, state_3, reward, done]
    output_next_state = output_y[:,:4]
    output_reward = output_y[:,4]
    output_done = output_y[:,5]
    
    # First and third env outputs are limited to +/- 2.4 and +/- 0.4
    output_next_state[:,0] = np.clip(output_next_state[:,0],-2.4,2.4)
    
    output_next_state[:,2] = np.clip(output_next_state[:,2],-0.4,0.4)
    
    # Threshold for being done is likliehood being > 0.01
    output_done = True if output_done > 0.01 or len(xs) > 500 else False
    
    return output_next_state, output_reward, output_done
    

tf.reset_default_graph()

num_hidden_m = 256

# Dimensions of the previous state plus 1 for the action
dimen_m = dimen + 1

# Placeholder for inputs
input_x_m = tf.placeholder(tf.float32, [None, dimen_m])

# First layer
W1_m = tf.get_variable("W1_m", shape=[dimen_m, num_hidden_m],
                     initializer=tf.contrib.layers.xavier_initializer())
B1_m = tf.Variable(tf.zeros([num_hidden_m]), name="B1M")
layer1_m = tf.nn.relu(tf.matmul(input_x_m, W1_m) + B1_m)

# Second layer
W2_m = tf.get_variable("W2_m", shape=[num_hidden_m, num_hidden_m],
                     initializer=tf.contrib.layers.xavier_initializer())
B2_m = tf.Variable(tf.zeros([num_hidden_m]), name="B2_m")
layer2_m = tf.nn.relu(tf.matmul(layer1_m, W2_m) + B2_m)

# Third (output) layers
# Note that there are three separate output layers, 
# one for next observation, reward and whether the game is complete

W_obs_m = tf.get_variable("W_obs_m", shape=[num_hidden_m, 4],
                     initializer=tf.contrib.layers.xavier_initializer())
B_obs_m = tf.Variable(tf.zeros([4]), name="B_obs_m")

W_reward_m = tf.get_variable("W_reward_m", shape=[num_hidden_m, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
B_reward_m = tf.Variable(tf.zeros([1]), name="B_reward_m")

W_done_m = tf.get_variable("W_done_m", shape=[num_hidden_m,1],
                     initializer=tf.contrib.layers.xavier_initializer())
B_done_m = tf.Variable(tf.zeros([1]), name="B_done_m")

output_obs_m = tf.matmul(layer2_m, W_obs_m) + B_obs_m
output_reward_m = tf.matmul(layer2_m, W_reward_m) + B_reward_m
output_done_m = tf.sigmoid(tf.matmul(layer2_m, W_done_m) + B_done_m)

# Placeholders for inputs used in training
actual_obs_m = tf.placeholder(tf.float32, [None, dimen_m], name="actual_obs")
actual_reward_m = tf.placeholder(tf.float32, [None, 1], name="actual_reward")
actual_done_m = tf.placeholder(tf.float32, [None, 1], name="actual_done")

# Putting it all together
predicted_state_m = tf.concat([output_obs_m, output_reward_m, output_done_m], axis=1)

# Loss functions
loss_obs_m = tf.square(actual_reward_m - output_reward_m)
loss_reward_m = tf.square(actual_reward_m - output_reward_m)
loss_done_m = -tf.log(actual_done_m * output_done_m + 
                (1 - actual_done_m) * (1 - output_done_m))

# Model loss is simply the average loss of the three outputs
loss_m = tf.reduce_max(loss_obs_m + loss_reward_m + loss_done_m)

adam_m = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_m = adam_m.minimize(loss_m)

num_hidden_p = 10 # Number of hidden units in the nn used to determine policy

input_x_p = tf.placeholder(tf.float32, [None, dimen], name="input_x")

# First layer
W1_p = tf.get_variable("W1", shape=[dimen,num_hidden_p], 
                     initializer=tf.contrib.layers.xavier_initializer())
layer1_p = tf.nn.relu(tf.matmul(input_x_p, W1_p))

# Second layer
W2_p = tf.get_variable("W2", shape=[num_hidden_p, 1], 
                     initializer=tf.contrib.layers.xavier_initializer())
output_p = tf.nn.sigmoid(tf.matmul(layer1_p, W2_p))

# Placeholders for inputs used in training
input_y_p = tf.placeholder(tf.float32, shape=[None, 1], name="input_y")
advantages_p = tf.placeholder(tf.float32, shape=[None,1], name="reward_signal")

# Loss function
# Below is equivalent to: 0 if input_y_p == output_p else 1
log_lik_p = tf.log(input_y_p * (input_y_p - output_p) + 
                 (1 - input_y_p) * (input_y_p + output_p))

# We'll be trying to maximize log liklihood
loss_p = -tf.reduce_mean(log_lik_p * advantages_p)

# Gradients
W1_grad_p = tf.placeholder(tf.float32,name="W1_grad")
W2_grad_p = tf.placeholder(tf.float32,name="W2_grad")
batch_grad_p = [W1_grad_p, W2_grad_p]
trainable_vars_p = [W1_p, W2_p]
grads_p = tf.gradients(loss_p, trainable_vars_p)

# Optimizer
adam_p = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Update function
update_grads_p = adam_p.apply_gradients(zip(batch_grad_p, [W1_p, W2_p]))

# Initialize and test to see models are setup correctly
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
random_obs = np.random.random(size=[1, env.observation_space.shape[0]])
random_action = env.action_space.sample()

print("obs: {}\naction: {}\noutput obs: {}\nouput reward: {}\noutput done: {}\noutput policy: {}".format(
        random_obs,
        random_action,
        sess.run(output_obs_m,feed_dict={input_x_m: np.hstack([random_obs, [[random_action]]])}),
        sess.run(output_reward_m,feed_dict={input_x_m: np.hstack([random_obs, [[random_action]]])}),
        sess.run(output_done_m,feed_dict={input_x_m: np.hstack([random_obs, [[random_action]]])}),
        sess.run(output_p,feed_dict={input_x_p: random_obs})))

# Tracks the score on the real (non-simulated) environment to determine when to stop
real_rewards = []
num_episodes = 5000

# Trigger used to decide whether we should train from model or from real environment
train_from_model = False
train_first_steps = 500

# Setup array to keep track of observations, rewards and actions
observations = np.empty(0).reshape(0,dimen)
rewards = np.empty(0).reshape(0,1)
actions = np.empty(0).reshape(0,1)

# Gradients
grads = np.array([np.zeros(var.get_shape().as_list()) for var in trainable_vars_p])

num_episode = 0

observation = env.reset()

while num_episode < num_episodes:
    observation = observation.reshape(1,-1)
    
    # Determine the policy
    policy = sess.run(output_p, feed_dict={input_x_p: observation})
    
    # Decide on an action based on the policy, allowing for some randomness
    action = 0 if policy > np.random.uniform() else 1

    # Keep track of the observations and actions
    observations = np.vstack([observations, observation])
    actions = np.vstack([actions, action])
    
    # Determine next observation either from model or real environment
    if train_from_model:
        observation, reward, done = step_model(sess, observations, action)
    else:
        observation, reward, done, _ = env.step(action)
        
    # Keep track of rewards
    rewards = np.vstack([rewards, reward])
    dones = np.zeros(shape=(len(observations),1))
    
    # If game is over or running long
    if done or len(observations) > 300:
        print("\r{} / {} ".format(num_episode, num_episodes),end="")

        # If we're not training our policy from our model, we'll train our model from the real env
        if not train_from_model:
             # Previous state and actions for training model
            states = np.hstack([observations, actions])
            prev_states = states[:-1,:]
            next_states = states[1:, :]
            next_rewards = rewards[1:, :]
            next_dones = dones[1:, :]

            feed_dict = {input_x_m: prev_states.astype(np.float32), 
                         actual_obs_m: next_states.astype(np.float32),
                        actual_done_m: next_dones.astype(np.float32),
                        actual_reward_m: next_rewards.astype(np.float32)}

            loss, _ = sess.run([loss_m, update_m], feed_dict=feed_dict)
            
            real_rewards.append(sum(rewards))
            
        
        # Discount rewards
        disc_rewards = discount(rewards, standardize=True)
        
        # Add gradients to running batch
        grads += sess.run(grads_p, feed_dict={input_x_p: observations,
                                            input_y_p: actions,
                                            advantages_p: disc_rewards})
        
        num_episode += 1
        
        observation = env.reset()

        # Reset everything
        observations = np.empty(0).reshape(0,dimen)
        rewards = np.empty(0).reshape(0,1)
        actions = np.empty(0).reshape(0,1)
        
        # Toggle between training from model and from real environment allowing sufficient time 
        # to train the model before its used for learning policy
        if num_episode > train_first_steps:
            train_from_model = not train_from_model 

        # If batch full
        if num_episode % policy_batch_size == 0:
            
            # Update gradients
            sess.run(update_grads_p, feed_dict={W1_grad_p: grads[0], W2_grad_p: grads[1]})
            
            # Reset gradients
            grads = np.array([np.zeros(var.get_shape().as_list()) for var in trainable_vars_p])
            
            # Print periodically
            if (num_episode % (100 * policy_batch_size) == 0):
                print("Episode {} last batch rewards: {}".format(
                        num_episode, sum(real_rewards[-policy_batch_size:])/policy_batch_size))
            
            # If our real score is good enough, quit
            if (sum(real_rewards[-10:]) / 10. >= 300):
                print("Episode {} Training complete with total score of: {}".format(
                        num_episode, sum(real_rewards[-policy_batch_size:])/policy_batch_size))
                break

# See our trained bot in action

observation = env.reset()
reward_sum = 0

model_losses = []

while True:
    env.render()
    
    observation = np.reshape(observation, [1, -1])
    policy = sess.run(output_p, feed_dict={input_x_p: observation})
    action = 0 if policy > 0.5 else 1
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    
    if done:
        print("Total score: {}".format(reward_sum))
        break



