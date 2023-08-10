get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simplejson as json
import os
from ASF import ASF, NIMBUS
from gradutil import *
from pyomo.opt import SolverFactory
from scipy.spatial.distance import euclidean
from interactiveBoreal import ReferenceFrame, Solver

nclust = 600
seedn = range(2,12)[4] # == 6

xses = init_norms()
x = xses['x']
x_norm = xses['x_norm']
x_stack = xses['x_stack']
x_norm_stack = xses['x_norm_stack']

ide = ideal(False)
nad = nadir(False)
opt = SolverFactory('cplex')
names = np.array(['Revenue', 'Carbon', 'Deadwood', 'Habitat'])
reg_names = np.array(['BAU','SA', 'EXT10','EXT30','GTR30','NTSR','NTL'])

with open('clusterings/new_{}.json'.format(nclust), 'r') as file:
    clustering = json.load(file)

c = np.array(clustering[str(seedn)]['c'])
xtoc = np.array(clustering[str(seedn)]['xtoc'])
dist = np.array(clustering[str(seedn)]['dist'])

weights = np.array([sum(xtoc == i) for i in range(nclust) if sum(xtoc == i) > 0])
indices = [min(np.array(range(len(xtoc)))[xtoc == i],
               key=lambda index: euclidean(x_norm[index], 
                                           np.mean(x_norm[xtoc == i],
                                                   axis=0)))
           for i in range(nclust) if sum(xtoc == i) > 0]

c_close = x_norm_stack[indices]
x_close = x_stack[indices]

data = c_close
nobj = np.shape(data)[-1]
nvar = len(x_norm)
w = weights/nvar
solver = SolverFactory('cplex')
problems = []
ress = []

for i in range(nobj):
    problems.append(BorealWeightedProblem(data[:, :, i], w, nvar))
    
for p in problems:
    ress.append(solver.solve(p.model))
    
payoff = [[cluster_to_value(x_close[:,:,i], res_to_list(p.model), weights) for i in range(nobj)] for p in problems]
ide_surr = np.max(payoff, axis=0)
nad_surr = np.min(payoff, axis=0)

payoff_model = [[model_to_real_values(x_stack[:, :, i], p.model, xtoc) for i in range(nobj)] for p in problems]
ide_orig = np.max(payoff_model, axis=0)
nad_orig = np.min(payoff_model, axis=0)

for p in payoff:
    for f in p:
        print('{:11.1f}'.format(f), end=' ')
    print('')

ide_surr, ide_orig, ide

(ide_surr-ide)/ide*100

nad_surr, nad_orig, nad

(nad_surr-nad)/nad*100

(nad_surr-nad)/ide*100

listss = [res_to_list(pro.model) for pro in problems]
revenue,_,_,_ = init_boreal()
orig_stands = revenue.values
all_regs = []
for l in listss:
    nos = dict()
    for ind,n in enumerate(l):
        these_stands_to_bau = orig_stands[xtoc==ind,int(n)]
        to_bau_no = np.sum(np.isnan(these_stands_to_bau))
        nos[0] = nos.get(0, 0) + to_bau_no
        nos[int(n)] = nos.get(n, 0) + weights[ind] - to_bau_no
    all_regs.append(nos)

for i,di in enumerate(all_regs):
    print('\n{}'.format(names[i]))
    summ = 0
    for key in di.keys():
        summ += di[key]
        print('{:5} {:6}'.format(reg_names[key], di[key]))
    print('Total: {}'.format(summ))
        

with open('optimizations/new_{}.json'.format(nclust), 'r') as file:
    optimization = json.load(file)

names = np.array(('revenue','carbon','deadwood','ha'))

def get_surr(name):
    return float(optimization[str(seedn)][name]['surrogate'])
optims_surr = np.array([get_surr(na) for na in names])

def get_orig(name):
    return float(optimization[str(seedn)][name]['real'])
optims_orig = np.array([get_orig(na) for na in names])

def ide_nad_normalize(point):
    return (point-nad)/(ide-nad)

pylab.rcParams['figure.figsize'] = (15,12)
from matplotlib import ticker

x = [1,2,3,4] # spines
y1 = ide_nad_normalize(ide_surr)
#y2 = ide_nad_normalize(ide_orig)
y3 = ide_nad_normalize(ide)
y4 = ide_nad_normalize(nad)

z1 = ide_nad_normalize(nad_surr)
#z2 = ide_nad_normalize(nad_orig)

fig, (ax, ax2, ax3) = plt.subplots(1, 3, sharey=False)

fig.suptitle('Comparing ideals and nadirs of the surrogate to the real ones.',
            fontsize=20)
color_surr = 'xkcd:bright red'
color_orig = 'xkcd:royal blue'
color_real = 'xkcd:apple green'

# plot the same on all the subplots
ax.plot( x, y1, color_surr, x, z1, color_surr , x, y3, color_real, x, y4, color_real)
ax2.plot(x, y1, color_surr, x, z1, color_surr , x, y3, color_real, x, y4, color_real)
ax3.plot(x, y1, color_surr, x, z1, color_surr , x, y3, color_real, x, y4, color_real)

'''
ax.plot( x, y2, color_orig, x, z2, color_orig)
ax2.plot(x, y2, color_orig, x, z2, color_orig)
ax3.plot(x, y2, color_orig, x, z2, color_orig)
'''

# now zoom in each of the subplots 
ax.set_xlim([x[0], x[1]])
ax2.set_xlim([x[1], x[2]])
ax3.set_xlim([x[2], x[3]])
ymin = -0.1
ymax = 1.25
ax.set_ylim( ymin, ymax)
ax2.set_ylim(ymin, ymax)
ax3.set_ylim(ymin, ymax)

# set the x axis ticks 
for axx, xx in zip([ax, ax2, ax3], x[:-1]):
    axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
    axx.xaxis.set_ticklabels([names[xx-1],names[xx]])
    axx.xaxis.set_tick_params(labelsize=15)
ax3.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))  # the last one
# add the labels to the rightmost spine
for tick in ax3.yaxis.get_major_ticks():
  tick.label2On=True

# stack the subplots together
plt.subplots_adjust(wspace=0)
plt.show()

