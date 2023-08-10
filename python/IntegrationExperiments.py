import functools
import itertools
import math

import matplotlib
from matplotlib import pyplot
import numpy as np
import scipy.integrate

import sys
sys.path.append("..")
from hiora_cartpole import fourier_fa
from hiora_cartpole import fourier_fa_int
from hiora_cartpole import offswitch_hfa
from hiora_cartpole import linfa
from hiora_cartpole import driver
from hiora_cartpole import interruptibility

import gym_ext.tools as gym_tools

import gym

def make_CartPole():
    return gym.make("CartPole-v0")

clipped_high = np.array([2.5, 3.6, 0.28, 3.7])
clipped_low  = -clipped_high
state_ranges = np.array([clipped_low, clipped_high])

four_n_weights, four_feature_vec     = fourier_fa.make_feature_vec(state_ranges,
                                  n_acts=2,
                                  order=3)

def make_uninterruptable_experience(choose_action=linfa.choose_action_Sarsa):
    return linfa.init(lmbda=0.9,
                        init_alpha=0.001,
                        epsi=0.1,
                        feature_vec=four_feature_vec,
                        n_weights=four_n_weights,
                        act_space=env.action_space,
                        theta=None,
                        is_use_alpha_bounds=True,
                        map_obs=functools.partial(gym_tools.warning_clip_obs, ranges=state_ranges),
                        choose_action=choose_action)

env         = make_CartPole()
fexperience = make_uninterruptable_experience()
fexperience, steps_per_episode, alpha_per_episode     = driver.train(env, linfa, fexperience, n_episodes=400, max_steps=500, is_render=False)
# Credits: http://matplotlib.org/examples/api/two_scales.html
fig, ax1 = pyplot.subplots()
ax1.plot(steps_per_episode, color='b')
ax2 = ax1.twinx()
ax2.plot(alpha_per_episode, color='r')
pyplot.show()

sr = state_ranges

def Q_at_x(e, x, a):
    return scipy.integrate.tplquad(
            lambda x_dot, theta, theta_dot: \
                e.feature_vec(np.array([x, x_dot, theta, theta_dot]), a)\
                    .dot(e.theta),
            sr[0][1],
            sr[1][1],
            lambda _: sr[0][2],
            lambda _: sr[1][2],
            lambda _, _1: sr[0][3],
            lambda _, _1: sr[1][3])

from multiprocessing import Pool
p = Pool(4)

def Q_fun(x):
    return Q_at_x(fexperience, x, 0)

num_Qs = np.array( map(Q_fun, np.arange(-2.38, 2.5, 0.5*1.19)) )
num_Qs

sym_Q_s0 = fourier_fa_int.make_sym_Q_s0(state_ranges, 3)

sym_Qs = np.array( [sym_Q_s0(fexperience.theta, 0, s0) 
                         for s0 in np.arange(-2.38, 2.5, 0.5*1.19)] )
sym_Qs

num_Qs[:,0] / sym_Qs

num_Qs[:,0] - sym_Qs

np.prod(state_ranges[1,1:] - state_ranges[0,1:])

mc_env = gym.make("MountainCar-v0")

mc_n_weights, mc_feature_vec = fourier_fa.make_feature_vec(
                                np.array([mc_env.low, mc_env.high]),
                                n_acts=3,
                                order=2)

mc_experience = linfa.init(lmbda=0.9,
                        init_alpha=1.0,
                        epsi=0.1,
                        feature_vec=mc_feature_vec,
                        n_weights=mc_n_weights,
                        act_space=mc_env.action_space,
                        theta=None,
                        is_use_alpha_bounds=True)

mc_experience, mc_spe, mc_ape = driver.train(mc_env, linfa, mc_experience,
                                            n_episodes=400,
                                            max_steps=200,
                                            is_render=False)

fig, ax1 = pyplot.subplots()
ax1.plot(mc_spe, color='b')
ax2 = ax1.twinx()
ax2.plot(mc_ape, color='r')
pyplot.show()

def mc_Q_at_x(e, x, a):
    return scipy.integrate.quad(
        lambda x_dot: e.feature_vec(np.array([x, x_dot]), a).dot(e.theta),
        mc_env.low[1],
        mc_env.high[1])

def mc_Q_fun(x):
    return mc_Q_at_x(mc_experience, x, 0)

sample_xs = np.arange(mc_env.low[0], mc_env.high[0], 
                      (mc_env.high[0] - mc_env.low[0]) / 8.0)

mc_num_Qs = np.array( map(mc_Q_fun, sample_xs) )
mc_num_Qs

mc_sym_Q_s0 = fourier_fa_int.make_sym_Q_s0(
                    np.array([mc_env.low, mc_env.high]),
                    2)

mc_sym_Qs = np.array( [mc_sym_Q_s0(mc_experience.theta, 0, s0)
                          for s0 in sample_xs] )
mc_sym_Qs 

mc_sym_Qs - mc_num_Qs[:,0]

# Credits: http://stackoverflow.com/a/1409496/5091738
def make_integrand(feature_vec, theta, s0, n_dim):
    argstr = ", ".join(["s{}".format(i) for i in xrange(1, n_dim)])
    
    code = "def integrand({argstr}):\n"            "    return feature_vec(np.array([s0, {argstr}]), 0).dot(theta)\n"                 .format(argstr=argstr)
            
    #print code
            
    compiled = compile(code, "fakesource", "exec")
    fakeglobals = {'feature_vec': feature_vec, 'theta': theta, 's0': s0,
                   'np': np}
    fakelocals = {}
    eval(compiled, fakeglobals, fakelocals)
    
    return fakelocals['integrand']

print make_integrand(None, None, None, 4)

for order in xrange(1,3):
    for n_dim in xrange(2, 4):
        print "\norder {} dims {}".format(order, n_dim)
        
        min_max = np.array([np.zeros(n_dim), 3 * np.ones(n_dim)])
        n_weights, feature_vec = fourier_fa.make_feature_vec(
                                    min_max,
                                    n_acts=1,
                                    order=order) 
        
        theta = np.cos(np.arange(0, 2*np.pi, 2*np.pi/n_weights))
        
        sample_xs = np.arange(0, 3, 0.3)
        
        def num_Q_at_x(s0):
            integrand = make_integrand(feature_vec, theta, s0, n_dim)
            return scipy.integrate.nquad(integrand,  min_max.T[1:])
        
        num_Qs =  np.array( map(num_Q_at_x, sample_xs) )
        #print num_Qs
        
        sym_Q_at_x = fourier_fa_int.make_sym_Q_s0(min_max, order)
        
        sym_Qs = np.array( [sym_Q_at_x(theta, 0, s0) for s0 in sample_xs] )
        #print sym_Qs
        
        print sym_Qs / num_Qs[:,0]

np.arange(0, 1, 10)

import sympy as sp
a, b, x, f = sp.symbols("a b x f")

b_int = sp.Integral(1, (x, a, b))

sp.init_printing()

u_int = sp.Integral((1-a)/(b-a), (x, 0, 1))

u_int

(b_int / u_int).simplify()

b_int.subs([(a,0), (b,2)]).doit()

u_int.subs([(a,0), (b,2)]).doit()

(u_int.doit()*b).simplify()



