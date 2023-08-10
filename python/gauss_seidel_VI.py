import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Gauss Seidel Variation to the Value Iteration method
    """
    
    def one_step_lookahead(state, V,V_old):
       
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
            
                ####### Gauss Seidel #######
                if next_state < state:
                    A[a] += prob * (reward + discount_factor * V[next_state])
                else:
                    A[a] += prob * (reward + discount_factor * V_old[next_state])
                        
                    
        return A
    
    V = np.zeros(env.nS)
    V_old = np.zeros(env.nS)
    
    t = 0

    while True:
        
        delta = 0
        V_old = V
        # For every state
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V,V_old)
            best_action_value = np.max(A)
            # Select best action
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Value function update
             
            V[s] = best_action_value        
        t +=1
        if delta < theta:
            break
        
    
    # Find optimal policy
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        
        A = one_step_lookahead(s, V, V_old)
        best_action = np.argmax(A)
        
        policy[s, best_action] = 1.0
    
    return policy, V,t

policy, v = value_iteration(env)[0],value_iteration(env)[1]

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

print("Converged in number of steps:")
print(value_iteration(env)[2])
print("")

