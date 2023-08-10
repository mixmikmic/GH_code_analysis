import numpy as np
from hmmlearn import hmm
import operator

np.random.seed(777)

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.8]) # started observation on a sunny day

transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])

emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

# predict a sequence of hidden states based on visible states
# Bob says: walk, clean, shop, shop, clean, walk
bob_says = np.array([[0, 2, 1, 1, 2, 0]]).T

def get_obs_prob_for_hidden_state(observation, hidden_state):
    # observation cound be 0, 1, 2
    # hidden_state could be 0, 1
    return emission_probability[hidden_state][observation]

def get_hidden_state_prob_given_prev_state(hidden_state, prev_hidden_state=None):
    if prev_hidden_state is None:
        return start_probability[hidden_state]
    return transition_probability[prev_hidden_state][hidden_state]

def get_hidden_state_prob(hidden_state, observation_step):
    if observation_step == 0:
        return start_probability[hidden_state]
    prob_score = 0
    for state_id in range(n_states):
        prob_score += get_hidden_state_prob_given_prev_state(hidden_state, state_id) *                           get_hidden_state_prob(state_id, observation_step - 1)
    return prob_score

def get_observation_prob(observation, observation_step):
    prob_score = 0
    for state_id in range(n_states):
        prob_score += get_obs_prob_for_hidden_state(observation, state_id) *                          get_hidden_state_prob(state_id, observation_step)
    return prob_score

def get_hidden_state_prob_by_observation(observation, state_id, observation_step):
    obs_prob_for_state = get_obs_prob_for_hidden_state(observation, state_id)
    state_prob = get_hidden_state_prob(state_id, observation_step)
    obs_prob = get_observation_prob(observation, observation_step)
    return obs_prob_for_state * state_prob/ obs_prob

get_hidden_state_prob_by_observation(0, 0, 2)

def get_path_score_by_states(observed_values, debug=False):
    state_probs = [0 for s_id, state in enumerate(states)]
    if len(observed_values) == 1:
        prev_scores = [1]
    else:
        prev_scores = get_path_score_by_states(observed_values[:-1], debug)
    curr_observation = observed_values[-1]
    for s_id_prev, prev_score in enumerate(prev_scores):
        score = 0
        if len(observed_values) == 1:
            s_id_prev = None 
        for s_id, state in enumerate(states):
            state_prob = get_hidden_state_prob_given_prev_state(s_id, s_id_prev)
            obs_prob = get_obs_prob_for_hidden_state(curr_observation, s_id)
            if debug: print('s_id_prev, s_id, prev_score, state_prob, obs_prob', s_id_prev, s_id, prev_score, state_prob, obs_prob)
            score = state_prob * obs_prob * prev_score
            state_probs[s_id] += score
    if debug: print('state_probs', observed_values, ':', state_probs)
    return state_probs

def get_path_score(observed_values, debug=False):
    return np.log(sum(get_path_score_by_states(observed_values, debug)))

# just observation scores
for v in range(len(bob_says.T[0])):
    print(v, bob_says.T[0][:v+1], get_path_score(bob_says.T[0][:v+1]))

get_path_score(bob_says.T[0])

import itertools

# observed_values = bob_says.T[0]

def get_all_possible_path_score_by_states(observed_values, debug=False):
    path_state_probs = {}
    if len(observed_values) == 1:
        prev_path_state_probs = {'': 0}
    else:
        prev_path_state_probs = get_all_possible_path_score_by_states(observed_values[:-1], debug)
    curr_observation = observed_values[-1]
    for prev_path, prev_score in prev_path_state_probs.items():
        score = 0
        if len(prev_path) == 0:
            s_id_prev = None 
        else:
            s_id_prev = int(prev_path[-1])
        for s_id, state in enumerate(states):
            new_path = prev_path + str(s_id)
            state_prob = get_hidden_state_prob_given_prev_state(s_id, s_id_prev)
            obs_prob = get_obs_prob_for_hidden_state(curr_observation, s_id)
            if debug: print('s_id_prev, s_id, prev_score, state_prob, obs_prob', s_id_prev, s_id, prev_score, state_prob, obs_prob)
            score = np.log(state_prob) + np.log(obs_prob) + prev_score
#             if debug: print('score', score, np.log(score))
#             new_score = np.log(score)
            path_state_probs[new_path] = score
    if debug: print('path_state_probs', path_state_probs)
    return path_state_probs

def get_best_path(observed_values, debug=False):
    res = get_all_possible_path_score_by_states(observed_values, debug)
    prob_states, prob_score = sorted(res.items(), key=lambda val: val[1], reverse=True)[0]
    return prob_states, prob_score

prob_states, prob_score = get_best_path(bob_says.T[0])

prob_states, prob_score

print("Bob says:", ", ".join(list(map(lambda x: observations[x], bob_says.T[0]))))
print("Alice hears:", ", ".join(list(map(lambda x: states[int(x)], list(prob_states)))))

from collections import defaultdict

def get_optimul_viterbi_path(observed_values, debug=False):
    state_path_prob = defaultdict(list)
    for s_id, state in enumerate(states):
        obs_prob = get_obs_prob_for_hidden_state(observed_values[0], s_id)
        state_path_prob[0].append({
            's_id': s_id,
            'score': np.log(get_hidden_state_prob_given_prev_state(s_id, None)) + np.log(obs_prob),
            'path': str(s_id)
        })
    for idx, obs_val in enumerate(observed_values[1:]):
        for s_id, state in enumerate(states):
            new_state_id = None
            new_state_score = None
            path_so_far = None
            obs_prob = get_obs_prob_for_hidden_state(obs_val, s_id)
            
            for s_id_prev, _ in enumerate(states):
                prev_score = state_path_prob[idx][s_id_prev]['score']
                state_prob = get_hidden_state_prob_given_prev_state(s_id, s_id_prev)
                score = np.log(state_prob) + np.log(obs_prob) + prev_score
                if debug: print('s_id_prev, s_id, prev_score, state_prob, obs_prob, score', s_id_prev, s_id, prev_score, state_prob, obs_prob, score)
                if new_state_id is None or score > new_state_score:
                    new_state_id = s_id
                    new_state_score = score
                    path_so_far = state_path_prob[idx][s_id_prev]['path']
            state_path_prob[idx+1].append({
                's_id': new_state_id,
                'score': new_state_score,
                'path': path_so_far + str(new_state_id)
            })
            if debug: print('idx, s_id, score, path', idx, new_state_id, new_state_score, path_so_far + str(new_state_id))
        if debug: print("--")
    return state_path_prob

def get_viterbi_path(observed_values, debug=False):
    res = get_optimul_viterbi_path(observed_values, debug)
    final_res = sorted(res[max(res)], key=lambda val: val['score'], reverse=True)[0]
    return final_res['path'], final_res['score']

path, score = get_viterbi_path(bob_says.T[0])

path, score

# Most probable state transition for the given observations
res = get_optimul_viterbi_path(bob_says.T[0])
final_res = sorted(res[max(res)], key=lambda val: val['score'], reverse=True)[0]

for idx, state in enumerate(final_res['path']):
    for path_vals in res[idx]:
        if list(path_vals['path'])[-1] == state:
            print(idx, bob_says.T[0][0:idx], path_vals['path'], path_vals['score'])
            break

print("Bob says:", ", ".join(list(map(lambda x: observations[x], bob_says.T[0]))))
print("Alice hears:", ", ".join(list(map(lambda x: states[int(x)], list(path)))))

model = hmm.MultinomialHMM(n_components=n_states, random_state=777)

model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

model._check()

logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")

logprob, alice_hears

print("Bob says:", ", ".join(list(map(lambda x: observations[x], bob_says.T[0]))))
print("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))

# Most probable state transition for the given observations
for v in range(len(bob_says.T[0])):
    logprob, alice_hears = model.decode(bob_says[:v+1], algorithm="viterbi")
    print(v, bob_says.T[0][:v+1], logprob, alice_hears)

# just observation scores
for v in range(len(bob_says.T[0])):
    print(v, bob_says.T[0][:v+1], model.score([bob_says.T[0][:v+1]]))

model.score([bob_says.T[0]])



