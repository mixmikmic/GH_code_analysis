import copy
from itertools import product
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mdp_lib.domains.gridworld import GridWorld
from planninginbeliefmodel import PlanningInObserverBeliefModel
from mdp_lib.domains.gridworldvis import visualize_trajectory, plot_text
from task import mdp_codes, mdp_params
from mdp_lib.util import sample_prob_dict
from util import mdp_to_feature_belief

np.random.seed(128374027)

#doing model parameters
do_discount = .99
do_softmax = .08

#showing model parameters
base_discount_rate = .99
base_softmax_temp = 3.0
obmdp_discount_rate = .9
obmdp_softmax_temp= 1

doing_models = []
seed_trajs = []
for p in mdp_params:
    p = copy.deepcopy(p)
    p['discount_rate'] = do_discount
    m = GridWorld(**p)
    m.solve()
    doing_models.append(m)
    
    #generate seed trajectories for OBMDP discretization
    for _ in xrange(20):
        traj = m.run(temp=.7)
        if traj[-1][1] != '%':
            continue
        seed_trajs.append([(w, a) for w, a, _, _ in traj])
        
with open("./cached_values/seed_trajs.pkl", 'wb') as f:
    pickle.dump(seed_trajs, f)

showing_models = []
tf = None
for i, rfc in enumerate(mdp_codes):
    starttime = time.time()
    print rfc,
    m = PlanningInObserverBeliefModel(
        base_discount_rate = base_discount_rate,
        base_softmax_temp = base_softmax_temp,
        obmdp_discount_rate = obmdp_discount_rate,
        obmdp_softmax_temp=obmdp_softmax_temp,
        
        true_mdp_code=rfc,
        discretized_tf=tf
    )
    m.seed_beliefs_with_trajs(seed_trajs)
    m.build()
    m.solve()
    showing_models.append(m.ob_mdp)
    tf = m.ob_mdp.get_discretized_tf()
    print " %.2fs" % (time.time() - starttime)

def calc_obs_sobs_traj(wtraj):
    b_sobs = np.array(showing_models[0].get_init_state()[0])
    s = showing_models[0].get_init_state()
    
    obs_traj = [s[0],]
    sobs_traj = [b_sobs,]
    for w, a in wtraj:
        # get next naive belief
        ns = showing_models[0].transition(s=s, a=a)
        obs_traj.append(ns[0])
        
        # calc next sophisticated belief
        show_a_probs = []
        for m in showing_models:
            a_probs = m.get_softmax_actionprobs(s=s, temp=obmdp_softmax_temp)
            show_a_probs.append(a_probs[a])
        show_a_probs = np.array(show_a_probs)
        b_sobs = b_sobs*show_a_probs
        b_sobs = b_sobs/np.sum(b_sobs)
        sobs_traj.append(b_sobs)
        
        s = ns
    return {'obs_traj': obs_traj, 'sobs_traj': sobs_traj}

def is_correct(row):
    rf = dict(zip(['orange', 'purple', 'cyan'], row['rf']))
    if rf[row['color']] == 'x'             and row['exp_safe'] < .5:
        return True
    elif rf[row['color']] == 'o'             and row['exp_safe'] >= .5:
        return True
    return False

def calc_correct_prob(row):
    rf = dict(zip(['orange', 'purple', 'cyan'], row['rf']))
    if rf[row['color']] == 'x':
        return 1 - row['exp_safe']
    elif rf[row['color']] == 'o':
        return row['exp_safe']

n_trajs = 100
forder = ['orange', 'purple', 'cyan']
model_obs_judgments = []
for mi, (do_m, show_m) in enumerate(zip(doing_models, showing_models)):
    do_wtrajs = []
    show_wtrajs = []
    
    print mi,
    starttime = time.time()
    for _ in xrange(n_trajs):
        # generate and interpret DOING trajectory
        do_traj = do_m.run(temp=do_softmax)
        do_traj = [(w, a) for w, a, nw, r in do_traj]
        
        belief_trajs = calc_obs_sobs_traj(do_traj)
        obs_judg = mdp_to_feature_belief(belief_trajs['obs_traj'][-1], mdp_codes, forder)
        obs_judg['rf'] = mdp_codes[mi]
        obs_judg['observer'] = 'naive'
        obs_judg['demonstrator'] = 'doing'
        obs_judg['traj'] = do_traj
        obs_judg['belief_traj'] = belief_trajs['obs_traj']
        model_obs_judgments.append(obs_judg)
        
        sobs_judg = mdp_to_feature_belief(belief_trajs['sobs_traj'][-1], mdp_codes, forder)
        sobs_judg['rf'] = mdp_codes[mi]
        sobs_judg['observer'] = 'sophisticated'
        sobs_judg['demonstrator'] = 'doing'
        sobs_judg['traj'] = do_traj
        sobs_judg['belief_traj'] = belief_trajs['sobs_traj']
        model_obs_judgments.append(sobs_judg)
        
        # generate and interpret SHOWING trajectory
        show_traj = show_m.run(temp=obmdp_softmax_temp)
        show_traj = [(w, a) for (b, w), a, ns, r in show_traj]
        
        belief_trajs = calc_obs_sobs_traj(show_traj)
        obs_judg = mdp_to_feature_belief(belief_trajs['obs_traj'][-1], mdp_codes, forder)
        obs_judg['rf'] = mdp_codes[mi]
        obs_judg['observer'] = 'naive'
        obs_judg['demonstrator'] = 'showing'
        obs_judg['traj'] = show_traj
        obs_judg['belief_traj'] = belief_trajs['obs_traj']
        model_obs_judgments.append(obs_judg)
        
        sobs_judg = mdp_to_feature_belief(belief_trajs['sobs_traj'][-1], mdp_codes, forder)
        sobs_judg['rf'] = mdp_codes[mi]
        sobs_judg['observer'] = 'sophisticated'
        sobs_judg['demonstrator'] = 'showing'
        sobs_judg['traj'] = show_traj
        sobs_judg['belief_traj'] = belief_trajs['sobs_traj']
        model_obs_judgments.append(sobs_judg)
    print " %.2fs" % (time.time() - starttime)
        
model_obs_judgments = pd.DataFrame(model_obs_judgments)
model_obs_judgments = pd.melt(model_obs_judgments,
    id_vars=['demonstrator', 'rf', 'observer', 'traj', 'belief_traj'], 
    value_name='exp_safe', 
    var_name='color')

model_obs_judgments['confidence'] = model_obs_judgments['exp_safe'].apply(lambda v: abs(.5-v))
model_obs_judgments['correct'] = model_obs_judgments.apply(is_correct, axis=1)
model_obs_judgments['correct_prob'] = model_obs_judgments.apply(calc_correct_prob, axis=1)

model_obs_judgments.to_pickle('./cached_values/model_obs_judgments.pkl')

