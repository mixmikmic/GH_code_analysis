import numpy as np

def compute_discounted_return(rewards, masks, gamma, next_value):
    """Compute discounted returns given a list of rewards, discount_factor, a list of termination masks, 
    and next value estiamte.
    
    """
    data_len = len(rewards) # get length of episode
    returns = np.zeros(data_len+1) # create returns, with additional index for next_value
    returns[-1] = next_value # set additional index as next_value
    
    for i in reversed(range(data_len)):
        returns[i] = rewards[i] + gamma*returns[i+1]*masks[i]
    # note that returns should exclude the additional index.
    return returns[:-1]
    
    
# CartPole Example
num_steps = 10
gamma = 1.0
rewards = np.ones(num_steps) # reward of all 1.0s
masks = np.ones(num_steps) # create our masks
masks[5] = 0.0 # set step 6 to be 0.0, indicating end of episode 1.
next_value_case_1 = 1.0 # next step is termination for episode 2.
next_value_case_2 = 10.0 # next step is not termination for episode 2.

returns_case_1 = compute_discounted_return(rewards, masks, gamma, next_value_case_1)
returns_case_2 = compute_discounted_return(rewards, masks, gamma, next_value_case_2)
print('return_case_1 is {} \nreturn_case_2 is {}'.format(returns_case_1, returns_case_2))



