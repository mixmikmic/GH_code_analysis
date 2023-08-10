import numpy as np
import itertools
import matplotlib.pyplot as plt

# Game environment
class Board:
    def __init__(self, p_random = 0.2):
        self.num_rows = 5
        self.num_cols = 5
        self.num_cells = self.num_cols * self.num_rows
        
        # Choose starting position of the agent randomly among the first 5 cells
        self.agent_position = np.random.randint(0, 5)
        
        # Choose position of the gold and bomb
        self.bomb_positions = np.array([18])
        self.gold_positions = np.array([23])
        self.terminal_states = np.array([self.bomb_positions, self.gold_positions])
       
        # Specify rewards
        self.rewards = np.zeros(self.num_cells)
        self.rewards[self.bomb_positions] = -10
        self.rewards[self.gold_positions] = 10
        
        # Specify available actions
        self.actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        
        self.p_random = p_random
    
    def get_available_actions(self):
        return self.actions
    
    def make_step(self, action): 
        old_position = self.agent_position
        new_position = self.agent_position
        
        # 0.2 probability to move the agent in other directions
        if np.random.rand(1) < self.p_random:
            all_actions = np.array(self.actions)
            available_actions = list(all_actions[all_actions!=action])
            action = available_actions[np.random.randint(0,len(available_actions))]
        
        # Update new_position based on the chosen action and check whether agent hits a wall.
        if action == "UP":
            candidate_position = self.agent_position + self.num_cols
            if candidate_position < self.num_cells:
                new_position = candidate_position
        elif action == "RIGHT":
            candidate_position = self.agent_position + 1
            if candidate_position % self.num_cols > 0:
                new_position = candidate_position
        elif action == "DOWN":
            candidate_position = self.agent_position - self.num_cols
            if candidate_position >= 0:
                new_position = candidate_position
        elif action == "LEFT": 
            candidate_position = self.agent_position - 1
            if candidate_position % self.num_cols < self.num_cols - 1:
                new_position = candidate_position
        else:
            raise ValueError('Action was mis-specified!')
        
        # Update the position of the agent.
        self.agent_position = new_position
        
        # Get reward 
        reward = self.rewards[new_position]
              
        # Deduct 1 from reward if agent moved
        if old_position != new_position:
            reward -= 1
        
        return reward
    
    # reset game when the game ended
    def reset_game(self):
        self.__init__()
        

# Random agent
class Random_Agent():
    def choose_action(self, available_actions):
        number_of_actions = len(available_actions)
        random_index = np.random.randint(0, number_of_actions)
        action = available_actions[random_index]
        return action
    

# Q learning agent
class Q_Agent:
    def __init__(self, alpha = 0.1, greedy = 0.05):
        self.alpha = alpha
        self.greedy = greedy
        self.q = {key: 0 for key in itertools.product(np.arange(25), ["UP", "RIGHT", "DOWN", "LEFT"])}
    
    def choose_action(self, available_actions, current_position):
        if np.random.rand(1) < self.greedy:
            # Choose not greedy action 
            number_of_actions = len(available_actions)
            index = np.random.randint(0, number_of_actions)
            action = available_actions[index]
        else:
            # Choose greedy action according to Q table
            # Get the maximum Q value of the current state
            max_q = np.array([self.q[key] for key in self.q if key[0]==current_position]).max()
            # Create a current dictionary which is a part of Q table to avoid of different state, but same q value
            current_key = [key for key in self.q if key[0]==current_position]
            new_d = {key:self.q[key] for key in current_key}
            # Choose the action, if there are several actions, choose randomly
            choices = [key for key in new_d if new_d[key]==max_q]
            action = choices[np.random.choice(len(choices))][1]
            
        return action
    
    def Q_learn(self, old_position, new_position, action, reward):
        # Update Q table
        max_value = np.array([self.q[key] for key in self.q if key[0]==new_position]).max()
        self.q[(old_position,action)] = (1-self.alpha)*self.q[(old_position,action)]+self.alpha*(reward+max_value)
        

# Train Q learning agent
def Train_Q(environment, agent, eposides):
    record = np.zeros(eposides)
    for eposide in range(eposides):
        total_reward = 0
        while environment.agent_position not in environment.terminal_states:
            old_position = environment.agent_position
            available_actions = environment.get_available_actions()
            chosen_action = agent.choose_action(available_actions,old_position)
            reward = environment.make_step(chosen_action)
            total_reward += reward
            new_position = environment.agent_position
            agent.Q_learn(old_position,new_position,chosen_action,reward)
        env.reset_game()
        record[eposide] = total_reward

    return record

# Random agent play the game
def Play_Random(environment, agent, eposides):
    record = np.zeros(eposides)
    for eposide in range(eposides):
        total_reward = 0
        while environment.agent_position not in environment.terminal_states:
            available_actions = environment.get_available_actions()
            chosen_action = agent.choose_action(available_actions)
            reward = environment.make_step(chosen_action)
            total_reward += reward
        env.reset_game()
        record[eposide] = total_reward
    
    return record

# Train and play the game
env = Board()
Q_agent = Q_Agent()
Random_agent = Random_Agent()
X = np.arange(1,501)
Q_y = Train_Q(env,Q_agent,500)
Random_y = Play_Random(env,Random_agent,500)

# Plot the curves

plt.plot(X, Q_y, c='r', label = 'Q Learning Agent')
plt.plot(X, Random_y, c='g', label = 'Random Agent')
plt.xlabel('Eposode')
plt.ylabel('Mean Reward')
plt.title('Learning Curve')
plt.legend(loc = 'best')
plt.show()

Q_agent.q



