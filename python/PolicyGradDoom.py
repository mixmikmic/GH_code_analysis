import tensorflow as tf
import numpy as np
import gym
import ppaquette_gym_doom
import matplotlib.pyplot as plt
import scipy.ndimage

env = gym.make('ppaquette/DoomHealthGathering-v0')

# Environment Parameters
n_actions = 3
n_epochs = 1500
n = 0
average = []
step = 1
batch_size = 5000
render = False

# Define our three actions of moving forward, turning left & turning right
choice = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

# Hyper Parameters
alpha = 1e-4
gamma = 0.99
normalize_r = True
save_path='models/healthGather.ckpt'
value_scale = 0.5
entropy_scale = 0.00
gradient_clip = 40

# Apply discount to episode rewards & normalize
def discount(r, gamma, normal):
    discount = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        discount[i] = G
    # Normalize 
    if normal:
        mean = np.mean(discount)
        std = np.std(discount)
        discount = (discount - mean) / (std)
    return discount

# Conv Layers
convs = [16,32]
kerns = [8,8]
strides = [4,4]
pads = 'VALID'
fc = 256
activation = tf.nn.elu

# Function for resizing image
def resize(image):
    # Greyscale Image
    x = np.mean(image,-1)
    # Normalize Pixel Values
    x = x/255
    x = scipy.misc.imresize(x, [84,84])
    return(x)

# Tensorflow Variables
X = tf.placeholder(tf.float32, (None,84,84,1), name='X')
Y = tf.placeholder(tf.int32, (None,), name='actions')
R = tf.placeholder(tf.float32, (None,), name='reward')
N = tf.placeholder(tf.float32, (None), name='episodes')
D_R = tf.placeholder(tf.float32, (None,), name='discounted_reward')

# Policy Network
# CONVOLUTION 1 - 1
with tf.name_scope('conv1'):
    filter1 = tf.truncated_normal([kerns[0], kerns[0], 1, convs[0]],
                                  dtype=tf.float32,
                                  stddev=1/np.sqrt(84**2))
    filter1 = tf.Variable(filter1, name='weights1')
    stride = [1,strides[0],strides[0],1]
    conv1 = tf.nn.conv2d(X, filter1, stride, padding=pads)
    biases1 = tf.Variable(tf.constant(0.0, shape=[convs[0]], dtype=tf.float32), name='biases1')
    out = tf.nn.bias_add(conv1, biases1)
    out1 = activation(out)
    
    
# CONVOLUTION 1 - 2
with tf.name_scope('conv2'):
    shape = int(np.prod(out1.get_shape()[1:]))
    filter2 = tf.truncated_normal([kerns[1], kerns[1], convs[0], convs[1]],
                                  dtype=tf.float32,
                                  stddev=1/np.sqrt(shape))
    filter2 = tf.Variable(filter2, name='weights2')
    stride = [1,strides[1],strides[1],1]
    conv2 = tf.nn.conv2d(out1, filter2, stride, padding=pads)
    biases2 = tf.Variable(tf.constant(0.0, shape=[convs[1]], dtype=tf.float32), name='biases2')
    out = tf.nn.bias_add(conv2, biases2)
    out2 = activation(out)
    

#FULLY CONNECTED
with tf.name_scope('fc1') as scope:
    shape = int(np.prod(out2.get_shape()[1:]))
    fc1w = tf.truncated_normal([shape, fc], 
                               dtype=tf.float32, 
                               stddev=1/np.sqrt(shape))
    fc1w = tf.Variable(fc1w, name='weights3')
    fc1b = tf.Variable(tf.constant(1.0, shape=[fc], dtype=tf.float32), name='biases3')
    flat = tf.reshape(out2, [-1, shape])
    out = tf.nn.bias_add(tf.matmul(flat, fc1w), fc1b)
    fc_1 = activation(out)
    

#POLICY FUNCTION
with tf.name_scope('policy') as scope:
    logitsW = tf.truncated_normal([fc, n_actions], 
                                  dtype=tf.float32, 
                                  stddev=1/np.sqrt(fc))
    logitsW = tf.Variable(logitsW, name='weights4')
    logitsB = tf.Variable(tf.constant(1.0, shape=[n_actions], dtype=tf.float32),
                       trainable=True, name='biases4')
    action_logits = tf.nn.bias_add(tf.matmul(fc_1, logitsW), logitsB)
    calc_action = tf.multinomial(action_logits, 1)
    aprob = tf.nn.softmax(action_logits)
    
#VALUE FUNCTION
with tf.name_scope('value') as scope:
    valueW = tf.truncated_normal([fc, 1], 
                                 dtype=tf.float32,
                                 stddev=1)
    valueW = tf.Variable(valueW, name='weights5')
    value = tf.matmul(fc_1, valueW)

tf.trainable_variables()

def rollout(batch_size, render):
    
    states, actions, rewards, rewardsFeed, discountedRewards = [], [], [], [], []
    state = resize(env.reset())
    episode_num = 0 
    action_repeat = 3
    reward = 0
    
    while True: 
        
        if render:
            env.render()
        
        # Run State Through Policy & Calculate Action
        feed = {X: state.reshape(1, 84, 84, 1)}
        action = sess.run(calc_action, feed_dict=feed)
        action = action[0][0]
        
        # Perform Action
        for i in range(action_repeat):
            state2, reward2, done, info = env.step(choice[action])
            reward += reward2
            if done:
                break
        
        # Store Results
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        
        # Update Current State
        reward = 0
        state = resize(state2)
        
        
        
        if done:
            
            episode_num += 1
            
            # Track Discounted Rewards
            rewardsFeed.append(rewards)
            discountedRewards.append(discount(rewards, gamma, normalize_r))
            
            if len(np.concatenate(rewardsFeed)) > batch_size:
                break
                
            # Reset Environment
            rewards = []
            state = resize(env.reset())
                         
    return np.stack(states), np.stack(actions), np.concatenate(rewardsFeed), np.concatenate(discountedRewards), episode_num

mean_reward = tf.divide(tf.reduce_sum(R), N)

# Define Losses
pg_loss = tf.reduce_mean((D_R - value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=Y))
value_loss = value_scale * tf.reduce_mean(tf.square(D_R - value))
entropy_loss = -entropy_scale * tf.reduce_sum(aprob * tf.exp(aprob))
loss = pg_loss + value_loss - entropy_loss

# Create Optimizer
optimizer = tf.train.AdamOptimizer(alpha)
grads = tf.gradients(loss, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, gradient_clip) # gradient clipping
grads_and_vars = list(zip(grads, tf.trainable_variables()))
train_op = optimizer.apply_gradients(grads_and_vars)

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tmp/dpg")
tf.summary.scalar('Total_Loss', loss)
tf.summary.scalar('PG_Loss', pg_loss)
tf.summary.scalar('Entropy_Loss', entropy_loss)
tf.summary.scalar('Value_Loss', value_loss)
tf.summary.scalar('Reward_Mean', mean_reward)
tf.summary.histogram('Conv1', filter1)
tf.summary.histogram('Conv2', filter2)
tf.summary.histogram('FC', fc1w)
tf.summary.histogram('Logits', logitsW)
tf.summary.histogram('Value', valueW)
write_op = tf.summary.merge_all()

# Load model if exists
saver = tf.train.Saver(tf.global_variables())
load_was_success = True 
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print("No saved model to load. Starting new session")
    writer.add_graph(sess.graph)
    load_was_success = False
else:
    print("Loaded Model: {}".format(load_path))
    saver = tf.train.Saver(tf.global_variables())
    step = int(load_path.split('-')[-1])+1

while step < n_epochs+1:
    # Gather Training Data
    print('Epoch', step)
    s, a, r, d_r, n = rollout(batch_size,render)
    mean_reward = np.sum(r)/n
    average.append(mean_reward)
    print('Training Episodes: {}  Average Reward: {:4.2f}  Total Average: {:4.2f}'.format(n, mean_reward, np.mean(average)))
          
    # Update Network
    sess.run(train_op, feed_dict={X:s.reshape(len(s),84,84,1), Y:a, D_R: d_r})
          
    # Write TF Summaries
    summary = sess.run(write_op, feed_dict={X:s.reshape(len(s),84,84,1), Y:a, D_R: d_r, R: r, N:n})
    writer.add_summary(summary, step)
    writer.flush()
          
    # Save Model
    if step % 10 == 0:
          print("SAVED MODEL")
          saver.save(sess, save_path, global_step=step)
          
    step += 1

state = resize(env.reset())
prob, val = sess.run([aprob, value], feed_dict={X: state.reshape(1, 84, 84, 1)})

print('Turn Right: {:4.2f}  Turn Left: {:4.2f}  Move forward {:4.2f}'.format(prob[0][0],prob[0][2], prob[0][1]))
print('Approximated State Value: {:4.4f}'.format(val[0][0]))



