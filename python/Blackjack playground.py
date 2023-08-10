import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../") 

from lib.envs.blackjack import BlackjackEnv

# Get the environment
env = BlackjackEnv()

# Number of episodes to run
n_episodes = 20

# Number of iterations to run
n_iter = 100

def print_observation(observation):
    """
    Print the observation in an interpretable format.
    
    """
    score, dealer_score, usable_ace = observation
    print('Player Score: {} (Usable ace: {}), Dealer Score: {}'.format(score, usable_ace, dealer_score))

def strategy(observation):
    """
    Decide the action to take for the given observation.
    
    """
    score, dealer_score, usable_ace = observation
    
    # Action 1: Show the next card
    # Action 0: Stop showing cards
    # If score >= 20, return action = 0, else action = 1
    return 0 if score >= 20 else 1

for i in range(n_episodes):
    observation = env.reset()
    
    for it in range(n_iter):
        
        print_observation(observation)
        action = strategy(observation)
        print("Taking action {}".format(["Stick", "Hit"][action]))
        observation, reward, done, _ = env.step(action)
        
        # Termination
        if done:
            print_observation(observation)
            print('Game ended. Reward = {} \n'.format(float(reward)))
            break



