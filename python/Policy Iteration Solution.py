import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: lambda discount factor.
    
    Returns: n개의 state에 대한 value function을 반환
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS) # number of States는 16(4x4의 gridWorld)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]): # policy는 16 x 4 (각 16 state에서 취할 수 있는 action 4개에 대한 확률)
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]: # state s에서 a으로 갔을 때의 environment 결과
                    # 다이나믹 프로그래밍은 P(전이확률)을 알고 있기 때문에 Bellman Equation으로 iterative하게 풀어낼 수 있다.
                    # expeced value를 계산
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # 그 전의 value와 theta보다 적은 차이를 보인다면 종료
        if delta < theta:
            break
    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA # policy는 16 x 4 (각 16 state에서 취할 수 있는 action 4개에 대한 확률)
    
    while True:
        # Policy evaluation 해서 16개 state의 expected value 받아온다
        V = policy_eval_fn(policy, env, discount_factor) 
        

        # policy를 바꾸길 원한다면 false로 해줘라
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s]) # 가장 확률이 높은 액션의 인덱스를 받아온다
            #(우리의 현재 policy 상 뽑힌 액션)
            
            # 한번 해보고 실제 지금 state에서 가장 높은 value를 주는 policy가 뭐였는지 찾아본다.
            # one-step ahead
            action_values = np.zeros(env.nA) # action value는 액션의 갯수 만큼..! (4개)
            for a in range(env.nA): # 이번엔 4개의 action에 대해서 expected return의 계산해본다.
                for prob, next_state, reward, done in env.P[s][a]: 
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values) # 한 스텝 해본 결과 가장 value가 높았던 action
            
            # Greedily update the policy
            if chosen_a != best_a: # 만약 현 policy가 정한 action과 실제 한 스텝 해봤을 때의 best action이 다르다면
                policy_stable = False # policy_stable은 False가 될테고
            
            # policy[s]에는 best_a로 업데이트한다. (s일 때 a 행동을 하도록 (discrete하게 one-hot으로 넣어줌))
            policy[s] = np.eye(env.nA)[best_a]
        
        # 만약 chosen_a와 best_a가 동일했다면 policy가 안정감 있다고 판단하고 종료한다.
        if policy_stable:
            return policy, V

policy, v = policy_improvement(env)
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


# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)



