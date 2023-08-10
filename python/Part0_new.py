import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
get_ipython().magic('matplotlib inline')

env = gym.make('CartPole-v0')

tf.reset_default_graph()

NUM_ACTIONS = env.action_space.n  # 0 or 1

n = 3

n_input = 4 * n
n_hidden_1 = 64
n_hidden_2 = 32
n_hidden_3 = 16
n_out = 2

weights = {
    'h1' : tf.Variable(tf.random_uniform([n_input, n_hidden_1], 0, 0.01)),
    'h2' : tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], 0, 0.01)),
    'h3' : tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3], 0, 0.01)),
    'out' : tf.Variable(tf.random_uniform([n_hidden_3, n_out], 0, 0.01))
}


def multilayer_model(x, weights):
    layer_1 = tf.matmul(x, weights['h1'])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']))
    layer_3 = tf.matmul(layer_2, weights['h3'])
    out_layer = tf.nn.softmax(tf.matmul(layer_3, weights['out']))
    return out_layer


inputs1 = tf.placeholder(shape=[None, n_input], dtype=tf.float32)

q_out = multilayer_model(inputs1, weights)
predict = tf.argmax(q_out, 1)

next_q = tf.placeholder(shape=[None, n_out], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_q - q_out))
update_model = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init = tf.initialize_all_variables()


discount_factor = 0.99
exploration_rate = 0.3
NUM_EPISODES = 500
MAX_T = 250
DEBUG = True


step_list = []
reward_list = []
prev_states = []
with tf.Session() as sess:
    sess.run(init)
    for episode in range(NUM_EPISODES):
        state_0 = env.reset()
        
        
        reward_all = 0
        t = 0
        while t < MAX_T:
            t += 1
            
            prev_states.append(state_0.reshape((1,4)))
            input_vec_0 = None
            ###############
            if len(prev_states) < n:
                input_vec_0 = np.zeros((1, 4 * (n-len(prev_states))))
                for i in range(len(prev_states)):
                    input_vec_0 = np.concatenate((input_vec_0, prev_states[i]), axis=1)
            else:
                for pr_st in prev_states[-n:]:
                    if input_vec_0 is None:
                        input_vec_0 = pr_st
                    else:
                        input_vec_0 = np.concatenate((input_vec_0, pr_st), axis=1)
            ###############
            
            action, all_q = sess.run([predict, q_out], feed_dict={inputs1:input_vec_0})
            if t > 300:
                exploration_rate = 0.001
                
            if np.random.rand(1) < exploration_rate:
                action[0] = env.action_space.sample()
            
            state, reward, done, _ = env.step(action[0])
            
            
            prev_states.append(state.reshape((1,4)))
            input_vec = None
            ###############
            if len(prev_states) < n:
                input_vec = np.zeros((1, 4 * (n-len(prev_states))))
                for i in range(len(prev_states)):
                    input_vec = np.concatenate((input_vec, prev_states[i]), axis=1)
            else:
                for pr_st in prev_states[-n:]:
                    if input_vec is None:
                        input_vec = pr_st
                    else:
                        input_vec = np.concatenate((input_vec, pr_st), axis=1)
            ###############
            
            q1 = sess.run(q_out, feed_dict={inputs1:input_vec})
    
            max_q1 = np.max(q1)
            
            target_q = all_q
            
            target_q[0, action[0]] = reward + discount_factor * max_q1
            
            _ = sess.run([update_model], feed_dict={inputs1:input_vec_0, next_q:target_q})
            
            reward_all += reward
            state_0 = state
            
            if done:
                if DEBUG:
                    print("Episode {} finished after {} timesteps".format(episode, t))
                break
                
        step_list.append(t)
        reward_list.append(reward_all)
        
print("Average score: {}".format(sum(reward_list)/NUM_EPISODES))
env.close()

plt.plot(reward_list)

def get_explore_rate(t):
    return max(0.1, min(1, 1.0 - math.log10((2*t+1)/25)))

t = np.linspace(0,100)
y = [get_explore_rate(i) for i in t]
plt.scatter(t, y)

