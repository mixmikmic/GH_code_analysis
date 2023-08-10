import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

tf.set_random_seed(3)  # 3 is my lucky number :D :)
np.random.seed(3)  # for setting a seed to the numpy sample behaviour

env = gym.make('CartPole-v0')  # create a lambda function for obtaining the environment

def play_one_cart_pole_episode(model, env, dis_gamma=0.9, render=False, ignore_done=False, max_steps=200):
    """
        play single episode of the cart-pole game in order to generate the learning data
        @args:
            model: neural network object used to predict the action
            dis_gamma: discount factor for calculating returns
        @returns:
            experience, episode_length => 
                (states, actions, returns): tuple of lists of state and return (**Not reward)
                length of the episode
    """
    # reset environment to obtain the first set of observations
    obs = env.reset()
    
    # initialize the states and rewards lists
    states = [obs]
    actions = []
    rewards = []  # note that initial state has no reward corresponding to it
    
    # play the game untill it lasts
    done = False
    steps = 1
    while not done and steps <= max_steps:
        # render if value is true
        if render:
            env.render()
        
        action_probs = model.predict_action(obs)
        action = np.random.choice(range(len(action_probs)), p=action_probs)
        
        # take the action on the environment
        obs, reward, done, _ = env.step(action)
        
        # append the state and reward to appropriate lists
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        
        if ignore_done:
            done = False
        
        steps += 1

    # remove the last element from the states list since last state is not required for learning
    states.pop(-1)
    
    episode_length = sum(rewards)
    
    # now that we have the rewards, calculate the returns from them
    # Note that return for a state is the 
    # Expected value of rewards (here onwards) discounted by the discount factor \gamma
    # G(t) = r + (gamma * G(t + 1))
    
    # initialize the returns list **Note that the last state has a return of 0
    returns = [0]
    
    # calculate the returns in reverse order since it is efficient to do so
    for reward in reversed(rewards):
        returns.append(reward + (dis_gamma * returns[-1]))
    
    # remove the initial 0 return from the list
    returns.pop(0)
    
    # reverse the returns list
    returns = list(reversed(returns))
        
    # ensure the lengths of states, actions and returns are equal
    assert len(states) == len(actions) and len(actions) == len(returns), "Computation messed up"
        
    # return the calculated lists
    return np.array(states), np.array(actions), np.array(returns), episode_length

class Model:
    """
        Neural Network model for cart-Pole task
    """
    
    def __create_graph(self, eps=1e-12):
        """ 
            private helper to create tensorflow graph 
        """
        
        graph = tf.Graph()
        
        with graph.as_default():
            # define the input placeholders for training the model
            with tf.name_scope("Inputs"):
                x = tf.placeholder(tf.float32, shape=(None, self.ip_dim), name="input")
                y = tf.placeholder(tf.int32, shape=(None,), name="taken_action")
                g = tf.placeholder(tf.float32, shape=(None,), name="returns")
    
            # convert y into one_hot values
            with tf.name_scope("One_Hot_Encoder"):
                y_one_hot = tf.one_hot(y, depth=self.num_classes, name="one_hot")
    
            # define the neural computations pipeline
            x_ = x
            count = 1
            for width in self.hl_widths:
                x_ = tf.layers.dense(x_, width, activation=tf.nn.tanh, name="Dense_"+str(count))
                count += 1
                
            # add the last dense layer for prediction
            y_ = tf.layers.dense(x_, self.num_classes, use_bias=False, name="Dense_"+str(count))
            
            
            # define the neural computations for the v_model pipeline
            x_ = x
            count = 1
            for width in self.v_model_hl_widths:
                x_ = tf.layers.dense(x_, width, activation=tf.nn.relu, name="V_Dense_"+str(count))
                count += 1
            # add the last v_model_layer
            y_vmod = tf.layers.dense(x_, 1, use_bias=False, name="V_Dense_"+str(count))
            
            # define the predictions block
            with tf.name_scope("Predictions"):
                predictions = tf.nn.softmax(y_)
                value_preds = y_vmod
                
            # define the losses for the graph
            with tf.name_scope("Losses"):
                loss = -tf.reduce_sum(((g - value_preds) * tf.log(tf.reduce_sum((predictions * y_one_hot), axis=-1) + eps)))
                v_loss = tf.reduce_sum(tf.square(g - value_preds))
                
            # define the optimizers
            with tf.name_scope("Opimizers"):
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(loss)
                v_optimizer = tf.train.AdagradOptimizer(learning_rate=self.vlr).minimize(v_loss)
                
            # define the init op
            with tf.name_scope("Errands"):
                init = tf.global_variables_initializer()
                
        # return the graph and important tensor handles
        return graph, x, y, g, predictions, loss, optimizer, value_preds, v_loss, v_optimizer, init
                
    
    def __init__(self, input_dim, num_classes, depth=1, hl_widths=[64], 
                 v_model_depth=1, v_model_hl_widths=[16], lr=3e-3, v_lr=3e-3):
        """
            create a dense Neural Network model including the value function approximating 
            Value model
            @args:
                input_dim: input dimensionality
                depth: number of hidden layers of the network. Note this 
                       doesn't include the last classification layer.
                hl_widths: list denoting the hidden layer widths of the network
                num_classes: final layer number of classes
                lr: learning rate 
        """
        
        # check if the depth and hl_widths are consistent
        assert depth == len(hl_widths), "hl_widths' length is not equal to depth"
        assert v_model_depth == len(v_model_hl_widths), "v_model_hl_widths' length is not equal to v_model_depth"
        
        # attach the values to the object data
        self.ip_dim = input_dim
        self.depth = depth
        self.hl_widths = hl_widths
        self.v_model_depth = v_model_depth
        self.v_model_hl_widths = v_model_hl_widths
        self.num_classes = num_classes
        self.lr = lr
        self.vlr = v_lr
        
        # create the graph and obtain the handles to tensors
        self.graph, self.x, self.y, self.g, self.pred, self.loss, self.opt, self.value_pred,         self.v_loss, self.v_opt, self.init = self.__create_graph()
        
        # attach an interactive session to the model object
        self.sess = tf.Session(graph=self.graph)
        
        # initialize the graph with random values
        self.sess.run(self.init)
        
    
    def predict_action(self, inp_obs):
        """
            make a prediction based on the input observation
            @args:
                inp_obs: a single input observation
            @returns:
                pred: probability distribution over the possible actions
        """
        gra_in = np.expand_dims(inp_obs, axis=0)  # add the batch axis
        
        # use the session to make the prediction
        pred = np.squeeze(self.sess.run(self.pred, feed_dict={self.x: gra_in}))
        
        # return the calculated predictions
        return pred
    
    
    def predict_value(self, inp_obs):
        """
            make a prediction for the value of the state based on the input observation
            @args:
                inp_obs: a single input observation
            @returns:
                value: value of the current state
        """
        gra_in = np.expand_dims(inp_obs, axis=0)  # add the batch axis
        
        # use the session to calculate the value
        value = np.squeeze(self.sess.run(self.value_pred, feed_dict={self.x: gra_in}))
        
        # return the calculated value
        return value
        
    
    def fit_policy(self, inp_obs, act_actions, returns, max_iter=300, convergence_thresh=1e-12, feed_back=True,
           feedback_factor=5):
        """
            fit the model on the givne data (input, action, return)
            @args:
                inp_obs: list of input observations
                act_actions: list of actions
                returns: list of obtained returns
        """
        # run the training until either convergence threshold is reached or max_iter are complete
        cnt = 1
        loss_delta = float('inf')
        prev_loss = 0
        while cnt <= max_iter and loss_delta > convergence_thresh:
            _, cur_loss = self.sess.run([self.opt, self.loss], feed_dict={
                                                                         self.x: inp_obs,
                                                                         self.y: act_actions,
                                                                         self.g: returns
                                                                     })
            if feed_back and cnt % (max_iter / feedback_factor) == 0:
                print("Current_step: ", cnt, "   Current_loss:", cur_loss)
            
            loss_delta = np.abs(cur_loss - prev_loss)
            prev_loss = cur_loss
            cnt += 1
            
        # print a message for training complete
        print("fit complete")
        return cur_loss
    
    
    def fit_value(self, inp_obs, returns, max_iter=300, convergence_thresh=1e-12, feed_back=True,
           feedback_factor=5):
        """
            fit the model on the givne data (input, action, return)
            @args:
                inp_obs: list of input observations
                returns: list of obtained returns
        """
        # run the training until either convergence threshold is reached or max_iter are complete
        cnt = 1
        loss_delta = float('inf')
        prev_loss = 0
        while cnt <= max_iter and loss_delta > convergence_thresh:
            _, cur_loss = self.sess.run([self.v_opt, self.v_loss], feed_dict={
                                                                         self.x: inp_obs,
                                                                         self.g: returns
                                                                     })
            if feed_back and cnt % (max_iter / feedback_factor) == 0:
                print("Current_step: ", cnt, "   Current_loss:", cur_loss)
            
            loss_delta = np.abs(cur_loss - prev_loss)
            prev_loss = cur_loss
            cnt += 1
            
        # print a message for training complete
        print("fit complete")
        return cur_loss

model = Model(4, 2, depth=2, hl_widths=[32, 32])  # 4 = num_of_inputs, 2 = num_actions

x, y, g, _ = play_one_cart_pole_episode(model, env)

model.fit_policy(x, y, g, max_iter=1000);

model.fit_value(x, y, max_iter=1000)

def train_network(env, policy_network, epochs_per_learn=5000, 
                  no_of_epochs=10, feedback_chk=2, value_train_per_n_epochs=5):
    """
        improve the policy by playing episodes
        @args:
            policy_network: model of the policy
            episodes_per_learn: no_of_episodes to play for one training_session
            epochs_per_learn: iterations per training_session
    """
    print("Training start")
    print("-------------------------------------------------------------------------------------------")
    
    losses = []; v_losses = []
    avg_episode_lengths = []
    
    for epoch in range(no_of_epochs):
        # generate data by playing episodes
        print("Epoch: ", epoch + 1)
        x_, y_, g_, ep_len = play_one_cart_pole_episode(policy_network, env)

        # perform training on this data
        print("fitting policy ... ")
        loss = policy_network.fit_policy(x_, y_, g_, max_iter=epochs_per_learn)
        
#         if (epoch + 1) %value_train_per_n_epochs == 0:
        print("fitting value ...")
        v_loss = policy_network.fit_value(x_, g_, max_iter=epochs_per_learn)
        v_losses.append(v_loss)
            
        if (epoch + 1) % feedback_chk == 0 or epoch == 0:
            # play an episode with render on
            play_one_cart_pole_episode(policy_network, env, render=True)
            
        losses.append(loss)
        avg_episode_lengths.append(ep_len)
            
        print("Average Episode Length: %d" %np.mean(ep_len))
    print("-------------------------------------------------------------------------------------------")
    print("Training complete")
    
    return losses, v_losses, avg_episode_lengths

env = gym.make('CartPole-v0')
losses, v_losses, avg_episode_lengths = train_network(env, model,
                                            no_of_epochs=500, feedback_chk=50, epochs_per_learn=30)

env.close()

# plot the losses and the avg episode lengths
plt.figure().suptitle("Objective Function")
plt.plot(losses)

plt.figure().suptitle("Value Approximation")
plt.plot(v_losses)

plt.figure().suptitle("Episode Lengths")
plt.plot(avg_episode_lengths)

env = gym.make('CartPole-v0')

play_one_cart_pole_episode(model, env, render=True);

play_one_cart_pole_episode(model, env, render=True, ignore_done=True, max_steps=400);

env.close()

