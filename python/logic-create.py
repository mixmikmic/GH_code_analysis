import numpy as np
from scipy.linalg import expm
import random

class move_gym():
    def __init__(self):
        self.scope = 2.0
        self.states = 6
        self.actions = 4
    def reset(self, s=[]):
        if np.array( s ).shape[0] == 0:
            self.obstacle_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            self.obstacle_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1

            self.move_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            self.move_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1  
            while ( abs(self.obstacle_x-self.move_x) < 1.05 ) & ( abs(self.obstacle_y-self.move_y) < 1.05 ):
                self.move_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
                self.move_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1      

            self.target_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            self.target_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
            while (  ( abs(self.obstacle_x-self.target_x) < 1.05 ) & ( abs(self.obstacle_y-self.target_y) < 1.05 )  ) |             (  ( abs(self.move_x-self.target_x) < 1.05 ) & ( abs(self.move_y-self.target_y) < 1.05 )  ):
                self.target_x = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1
                self.target_y = np.random.randint( int(-self.scope*10), int(self.scope*10) )*0.1        
        else:
            self.obstacle_x=s[0]
            self.obstacle_y=s[1]
            self.target_x=s[2]
            self.target_y=s[3]
            self.move_x=s[4]
            self.move_y=s[5]

        state=np.array([ self.obstacle_x, self.obstacle_y, self.target_x, self.target_y, self.move_x, self.move_y ])
        return state #, self.state2img(state)
    def step(self, action):
        velocity = 0.2
        if action==0: # up down right left
            self.move_y+=velocity
        if action==1:
            self.move_y-=velocity
        if action==2:
            self.move_x+=velocity
        if action==3:
            self.move_x-=velocity

        if self.move_x > (self.scope+1.0):
            self.move_x-=velocity
        if self.move_x < (-self.scope-1.0):
            self.move_x+=velocity
        if self.move_y > (self.scope+1.0):
            self.move_y-=velocity
        if self.move_y < (-self.scope-1.0):
            self.move_y+=velocity
        
        reward = -0.1
        done = False
        info = "^_^"
        if (  ( abs(self.obstacle_x-self.move_x) < 1.05 ) & ( abs(self.obstacle_y-self.move_y) < 1.05 )  ): 
            reward = -1.0 
            done = True
            info = "collision"

        elif (  ( abs(self.target_x-self.move_x) < 1.05 ) & ( abs(self.target_y-self.move_y) < 1.05 )  ):
            reward = 1.0
            done = True
            info = "reach"

        state=np.array([ self.obstacle_x, self.obstacle_y, self.target_x, self.target_y, self.move_x, self.move_y ])

        return state, reward,done,info
    def state2img(self, state):
        img = np.zeros([84,84,3], dtype=np.uint8)
        ( img[ int(42-(self.obstacle_y+0.5)*10):int(42-(self.obstacle_y-0.5)*10), int((self.obstacle_x-0.5)*10+42):int((self.obstacle_x+0.5)*10+42),2 ] ).fill(255)
        ( img[ int(42-(self.target_y+0.5)*10):int(42-(self.target_y-0.5)*10), int((self.target_x-0.5)*10+42):int((self.target_x+0.5)*10+42),1 ] ).fill(255)
        ( img[ int(42-(self.move_y+0.5)*10):int(42-(self.move_y-0.5)*10), int((self.move_x-0.5)*10+42):int((self.move_x+0.5)*10+42),0 ] ).fill(255)
        return img

import tensorflow as tf
import numpy as np
from numpy import *
import random
from collections import deque

from tqdm import trange

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init. some parameters
        self.epsilon = INITIAL_EPSILON
        
        # self.state_dim = env.observation_space.shape[0]
        self.state_dim = env.states

        # self.action_dim = env.action_space.n
        self.action_dim = env.actions

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        
#         # # a: create new network weights
#         print "create new network weights ..."
#         num_hid = 200
#         self.W1 = self.weight_variable([self.state_dim,num_hid])
#         self.b1 = self.bias_variable([num_hid])
#         self.W2 = self.weight_variable([num_hid,num_hid])
#         self.b2 = self.bias_variable([num_hid])
#         self.W3 = self.weight_variable([num_hid,self.action_dim])
#         self.b3= self.bias_variable([self.action_dim])

        # # b: restore old network weights
        print "restore old network weights ..."
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.restore_wb( "train-3500/" )


        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,self.W1) + self.b1)
        h_layer02 = tf.nn.relu(tf.matmul(h_layer,self.W2) + self.b2)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer02,self.W3) + self.b3

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
          self.y_input:y_batch,
          self.action_input:action_batch,
          self.state_input:state_batch
          })

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
          self.state_input:[state]
          })[0]
        if random.random() <= 0.5:
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def obtain_wb(self):
        W1=self.session.run(self.W1)
        b1=self.session.run(self.b1)
        W2=self.session.run(self.W2)
        b2=self.session.run(self.b2)
        W3=self.session.run(self.W3)
        b3=self.session.run(self.b3)
        return W1, b1, W2, b2, W3, b3

    def restore_wb(self, directory):
        W1 = np.load( directory+"W1.npy" )
        W2 = np.load( directory+"W2.npy" )
        W3 = np.load( directory+"W3.npy" )

        b1 = np.load( directory+"b1.npy" )
        b2 = np.load( directory+"b2.npy" )
        b3 = np.load( directory+"b3.npy" )        
        return tf.Variable(W1), tf.Variable(b1), tf.Variable(W2), tf.Variable(b2), tf.Variable(W3), tf.Variable(b3)
    def use_nn(self,state):
        return self.Q_value.eval(feed_dict = {
          self.state_input:[state]
          })

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
          self.state_input:[state]
          })[0])

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    
    
# Hyper Parameters
STEP = 50 # Step limitation in an episode
TEST = 10 # The number of experiment for test

# initialize env and agent
env = move_gym()
agent = DQN(env)

all_episode = 0
all_reward = []



def discrete_it(state, discrete):
    
    dis_s = []
    for i in xrange( len(state) ):
        num = state[i]
        over = num - discrete
        for j in xrange( len(over) ):
            if over[j]<0:
                over[j] = 100 # a certain big number
        dis_s.append( np.argmin( over ) )
    return dis_s

def state2relative(state):
    top_t = 100.0
    bottom_t = 100.0
    right_t = 100.0
    left_t = 100.0

    top_o = 100.0
    bottom_o = 100.0
    right_o = 100.0
    left_o = 100.0        

    obstacle_x=state[0]
    obstacle_y=state[1]
    target_x=state[2]
    target_y=state[3]
    move_x=state[4]
    move_y=state[5]

    if target_y >= move_y:
        top_t = target_y - move_y
    if target_y < move_y:
        bottom_t = move_y - target_y
    if target_x >= move_x:
        right_t = target_x - move_x
    if target_x < move_x:
        left_t = move_x - target_x

    if obstacle_y >= move_y:
        top_o = obstacle_y - move_y
    if obstacle_y < move_y:
        bottom_o = move_y - obstacle_y
    if obstacle_x >= move_x:
        right_o = obstacle_x - move_x
    if obstacle_x < move_x:
        left_o = move_x - obstacle_x

    return np.array([ top_t, bottom_t, right_t, left_t, top_o, bottom_o, right_o, left_o ])



def logic_create(discrete):
#     logic_q = np.zeros( [7,7,7,7   ,7,7,7,7   , 5] )
    logic_q = np.load('logic_q-1.02-A8.npy')
    
    for it_tms in trange(50):
        info ="^_^"
        good_state = False
        for i in xrange(3000):
            if (info=="reach") & (good_state == False):
                state = env.reset(A)
#                 C.append(A)
                good_state = True
            else:
                state = env.reset()
                good_state = False
            A = state
            for j in xrange(STEP):
                relative_state=state2relative(state)
                # discrete it:
                relative_state = discrete_it( relative_state, discrete )

                action = agent.action(state)
                if good_state:
                    if max(logic_q[relative_state[0]][relative_state[1]][relative_state[2]][relative_state[3]]                        [relative_state[4]][relative_state[5]][relative_state[6]][relative_state[7]])<1000:

                        logic_q[relative_state[0]][relative_state[1]][relative_state[2]][relative_state[3]]                        [relative_state[4]][relative_state[5]][relative_state[6]][relative_state[7]][action+1] +=1

                state, reward,done,info = env.step(action)
                if done:
                    break

        ###############################
        if it_tms == 49: #%10 == 9:
            B = 0
            B11= 0

            for i in xrange(1000):
                state = env.reset()
                A = state
                for j in xrange(STEP):
                    relative_state=state2relative(state)
                    # discrete it:
                    relative_state = discrete_it( relative_state, discrete )

                    action = np.argmax( logic_q[relative_state[0]][relative_state[1]][relative_state[2]][relative_state[3]]                    [relative_state[4]][relative_state[5]][relative_state[6]][relative_state[7]] )-1
                    state, reward,done,info = env.step(action)
                    if done:
                        if info=="collision":
#                             B1.append(A)
#                             B11.append(A)
                                B11 += 1
                        break
                if done == 0:
#                     B.append(A)
                        B += 1

        ###############################
    
#             print "len(B): ", len(B)
#             print "len(B11): ", len(B11)
#             print "len(B1): ", len(B1)
#             np.save("logic_q-only", logic_q)
#             np.save("C", C)
            print B, B11
    np.save('logic_q-1.02-A9', logic_q)
    return B, B11 # len(B), len(B11)

discrete = np.array( [0., 1.02, 1.86, 2.93, 4.1, 5.03] )


n_no_done, n_collision = logic_create(discrete)

print ">>: ", n_no_done, n_collision, discrete



