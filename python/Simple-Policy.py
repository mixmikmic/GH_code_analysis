import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import OpenVSPGame as VSP

#List out our bandit arms. 
#Currently arm 4 (index #3) is set to most often provide a positive reward.
bandit_arms = VSP.bandit_arms
num_arms = VSP.num_arms

tf.reset_default_graph()

#These two lines established the feed-forward part of the network. 
weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network.
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_output = tf.slice(output,action_holder,[1])
loss = -(tf.log(responsible_output)*reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)

total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_arms) #Set scoreboard for bandit arms to 0.

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        
        #Choose action according to Boltzmann distribution.
        actions = sess.run(output)
        a = np.random.choice(actions,p=actions)
        action = np.argmax(actions == a)

        reward = VSP.start_game(bandit_arms[action]) #Get our reward from picking one of the bandit arms.
        
        #Update the network.
        _,resp,ww = sess.run([update,responsible_output,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        
        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print "Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward)
        i+=1
print "\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promising...."
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
    print "...and it was right!"
else:
    print "...and it was wrong!"





