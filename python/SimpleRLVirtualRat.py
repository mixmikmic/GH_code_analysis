import numpy as npp
import minpy.numpy as np
import cPickle
import matplotlib.pyplot as plt
import minpy
minpy.set_global_policy("only_numpy")

from SimplePolicyNetwork import SimplePolicyNetwork
from SimpleRLPolicyGradientSolver import SimpleRLPolicyGradientSolver
from simpleBox import simpleBox
from SimRat import SimRat
from dataProcessFunctions import *

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (8.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

box = simpleBox(mode="alternative",length=30000,block_size=20,random_range=0,trial_per_episode=5, repeat = True)
val_X, val_y = box.X, box.y

box.change_mode("pro_only")

init_params = []
for i in range(20):
    print "Training number %d" % (i,)
    print "pro_only"
    model = SimplePolicyNetwork(hidden_dim=5,reg=0)
    solver = SimpleRLPolicyGradientSolver(model, box,
                                        update_rule='adam',
                                        optim_config={
                                            'learning_rate': 0.01,
                                            'decay_rate': 1
                                        },
                                        init_rule='xavier',
                                        num_episodes=150,
                                        verbose=False,
                                        supervised = False,
                                        print_every=100)
    solver.init()
    init_params.append(solver.save_params())
    solver.train()
    box.change_mode("alternative")
    print "Alternative"
    solver.change_settings(learning_rate=5e-3,num_episodes = 3000)
    solver.train()
    
    rat = SimRat(model)
    probs = rat.predict(val_X,val_y)
    ratname = 'VirtualRat'
    loss_history(solver, ratname)
    sample_probabilities(probs, ratname, sample = 100)
    sample_correct_rate(rat, sample = 100)
    trial_window = 3
    np.set_printoptions(precision=2)
    plt.ylim([0,1])
    draw_3d(rat.p2a_prob, rat.a2p_prob, trial_window = 3)

file_name = "good_weights/good_weights"
a = 0
suffix = ".pkl"
for i in [0,2,8,9]:
    a+=1
    save_weights(file_name+str(a)+suffix,init_params[i])
    

box = simpleBox(mode="alternative",length=10000,block_size=20,random_range=0,trial_per_episode=5, repeat = True)
val_X, val_y = box.X, box.y

box.change_mode("pro_only")

init_params = load_weights("good_weights/good_weights1.pkl")

model = SimplePolicyNetwork(hidden_dim=5,reg=0)
solver = SimpleRLPolicyGradientSolver(model, box,
                                    update_rule='adam',
                                    optim_config={
                                        'learning_rate': 0.01,
                                        'decay_rate': 1
                                    },
                                    init_rule='xavier',
                                    num_episodes=150,
                                    verbose=False,
                                    supervised = False,
                                    print_every=100)
solver.load_params(init_params)
solver.train()
box.change_mode("alternative")
print "Alternative"
solver.change_settings(learning_rate=5e-3,num_episodes = 3000)
solver.train()

rat = SimRat(model)
box.change_mode("alternative")
probs = rat.predict(box.X,box.y)
ratname = 'VirtualRat'
loss_history(solver, ratname)
sample_probabilities(probs, ratname, sample = 100)
sample_correct_rate(rat, sample = 100)
trial_window = 3
np.set_printoptions(precision=2)
plt.ylim([0,1])
draw_3d(rat.p2a_prob, rat.a2p_prob, trial_window = 3)

