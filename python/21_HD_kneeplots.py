get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (15,12)
import numpy as np
import simplejson as json
import os
from gradutil import *
ide = ideal(False)

new_optims = dict()
for nclust in range(50, 8301, 50):
    try:
        with open('optimizations/hope_{}.json'.format(nclust), 'r') as file:
            optimizations = json.load(file)
    except FileNotFoundError:
        break
    new_optims[nclust] = dict()
    for seedn in optimizations.keys():
        new_optims[nclust][eval(seedn)] = dict()
        for name in optimizations[seedn].keys():
            new_optims[nclust][eval(seedn)][name] = dict()
            for key in optimizations[seedn][name].keys():
                new_optims[nclust][eval(seedn)][name][key] = float(optimizations[seedn][name][key])

new_optims[100][2]

inds = []
real_revenue = []
real_carbon = []
real_deadwood = []
real_ha = []
surr_revenue = []
surr_carbon = []
surr_deadwood = []
surr_ha = []
for nclust in sorted(new_optims.keys()):
    r_rev = []
    r_car = []
    r_dea = []
    r_ha = []
    s_rev = []
    s_car = []
    s_dea = []
    s_ha = []
    for seedn in new_optims[nclust].keys():
        r_rev.append(new_optims[nclust][seedn]['revenue']['real'])
        r_car.append(new_optims[nclust][seedn]['carbon']['real'])
        r_dea.append(new_optims[nclust][seedn]['deadwood']['real'])
        r_ha.append(new_optims[nclust][seedn]['ha']['real'])
        s_rev.append(new_optims[nclust][seedn]['revenue']['surrogate'])
        s_car.append(new_optims[nclust][seedn]['carbon']['surrogate'])
        s_dea.append(new_optims[nclust][seedn]['deadwood']['surrogate'])
        s_ha.append(new_optims[nclust][seedn]['ha']['surrogate'])
    real_revenue.append(r_rev)
    real_carbon.append(r_car)
    real_deadwood.append(r_dea)
    real_ha.append(r_ha)
    surr_revenue.append(s_rev)
    surr_carbon.append(s_car)
    surr_deadwood.append(s_dea)
    surr_ha.append(s_ha)
    inds.append(nclust)
inds = np.array(inds)
real_revenue = np.array(real_revenue)
real_carbon = np.array(real_carbon)
real_deadwood = np.array(real_deadwood)
real_ha = np.array(real_ha)
surr_revenue = np.array(surr_revenue)
surr_carbon = np.array(surr_carbon)
surr_deadwood = np.array(surr_deadwood)
surr_ha = np.array(surr_ha)

pylab.rcParams['figure.figsize'] = (15,12)

fig, ax = plt.subplots(2,2)
fig.suptitle('Optimization results using values from previously formed clustering surrogate.\nValues from 10 independent runs',
            fontsize=20)

data = np.array([[surr_revenue, surr_carbon], [surr_deadwood, surr_ha]])
names = np.array([['Revenue', 'Carbon'],['Deadwood', 'Habitat']])
optims = np.array([ideal(False)[:2], ideal(False)[2:]])
ymins = np.array([[2.3e+8,4.2e+6],[1.9e+5, 1.8e+4]])
ymaxs = np.array([[2.55e+8,4.5e+6],[2.3e+5, 2.1e+4]])
for i in range(np.shape(ax)[0]):
    for j in range(np.shape(ax)[1]):
        ax[i,j].plot(inds, np.max(data[i,j], axis=1),color='g')
        ax[i,j].plot(inds, np.mean(data[i,j], axis=1), color='y')
        ax[i,j].plot(inds, np.min(data[i,j], axis=1), color='r')
        #ax[i,j].plot(inds, data[i,j][:,3])
        ax[i,j].plot((min(inds), max(inds)),(optims[i,j], optims[i,j]), color='b')
        ax[i,j].set_title(names[i,j], fontsize=15)
        ax[i,j].set_ylim(ymin=ymins[i,j], ymax=ymaxs[i,j])
        ax[i,j].set_xlabel('Number of clusters', fontsize=12)
        ax[i,j].set_ylabel('Optimization results', fontsize=12)
        #for k in range(200, 1401, 200):
            #ax[i,j].axvline(x=k, ymin=0, ymax=250)

surr_all_stack = np.dstack((surr_revenue, surr_carbon, surr_deadwood, surr_ha))

np.min(np.min(abs((surr_all_stack-ide)/ide), axis=1), axis=0)*100

pylab.rcParams['figure.figsize'] = (15,12)

fig, ax = plt.subplots(2,2)
fig.suptitle('Optimization results using original variable values\nwhen clustering based surrogate mapped to original variables.\nValues from 10 independent runs',
            fontsize=20)

data = np.array([[real_revenue, real_carbon], [real_deadwood, real_ha]])
for i in range(np.shape(ax)[0]):
    for j in range(np.shape(ax)[1]):
        ax[i,j].plot(inds, np.max(data[i,j], axis=1), color='g')
        ax[i,j].plot(inds, np.mean(data[i,j], axis=1), color='y')
        ax[i,j].plot(inds, np.min(data[i,j], axis=1), color='r')
        ax[i,j].plot((min(inds), max(inds)),(optims[i,j], optims[i,j]), color='b')
        ax[i,j].set_title(names[i,j], fontsize=15)
        ax[i,j].set_ylim(ymin=0, ymax=ymaxs[i,j])
        ax[i,j].set_xlabel('Number of clusters', fontsize=12)
        ax[i,j].set_ylabel('Optimization results', fontsize=12)
        #for k in range(200, 1401, 200):
         #   ax[i,j].axvline(x=k, ymin=0, ymax=250)

real_all_stack = np.dstack((real_revenue, real_carbon, real_deadwood, real_ha))

np.min(np.min(abs((real_all_stack-ide)/ide), axis=1), axis=0)*100

ide_reshape = (np.ones((4,len(surr_revenue)))*ide.reshape(4,1))
max_all = (np.array((np.max(surr_revenue, axis=1), np.max(surr_carbon, axis=1), np.max(surr_deadwood, axis=1), np.max(surr_ha, axis=1)))-ide_reshape)/ide_reshape

pylab.rcParams['figure.figsize'] = (10,8)
plt.suptitle('Relative differences in objectives', fontsize=15)
plt.plot(inds, max_all.transpose())
plt.plot(inds, sum([np.abs(num) for num in max_all.transpose()],axis=1), color='r')
plt.plot(inds, np.zeros(len(inds)))
plt.xlabel('Number of clusters', fontsize=12)
plt.ylabel('Relative differences', fontsize=12)
plt.axvline(x=1350, ymin=0, ymax=250)

np.shape(surr_all_stack)

sums_all = sum(abs((surr_all_stack[:20]-ide)/ide), axis=2)

nc = np.argmin([sums_all[i,n] for i,n in enumerate(np.argmin(sums_all, axis=1))])

sn = np.argmin(sums_all[nc])

inds[nc], range(2,12)[sn]

sums_all[nc,sn]

pylab.rcParams['figure.figsize'] = (10,8)
plt.suptitle('Sums of relative optimization errors of the four objectives,\nfor all the clusterings.\nUsing the proxy variable based results.', fontsize=15)
plt.scatter(inds[nc], sum(abs((surr_all_stack[nc,sn]-ide)/ide)), color='b', s=120)
plt.scatter(np.ones((len(surr_revenue),10))*inds.reshape(len(surr_revenue),1), sum(abs((surr_all_stack-ide)/ide), axis=2), color='r')
plt.xlabel('Number of clusters', fontsize=12)
plt.ylabel('Sums of relative differences', fontsize=12)
# plt.axvline(x=1500, ymin=0, ymax=250)

pylab.rcParams['figure.figsize'] = (10,15)

fig, ax = plt.subplots(2,1)

ran = range(2,12)
handles=[]
ax[0].set_title('Relative differences in objectives, {} clusters'.format(inds[nc]), fontsize=15)
for val,ie,name in zip(surr_all_stack[nc,].transpose(),ide,names.flatten()):
    handles.append(ax[0].scatter(ran, (val-ie)/ie, label=name))
ax[0].set_xticks(np.arange(min(ran), max(ran)+1, 1.0))
ax[0].set_xlabel('Clustering initialization seed', fontsize=12)
ax[0].set_ylabel('Relative differences', fontsize=12)
ax[0].legend(handles=handles, fontsize=12)

ax[1].set_title('Sum of absolute values of relative differences of all the objectives', fontsize=15)
ax[1].scatter(range(2,12), sum([abs(n) for n in (surr_all_stack[11]-ide)/ide], axis=1))
ax[1].set_xticks(np.arange(min(ran), max(ran)+1, 1.0))
ax[1].set_xlabel('Clustering initialization seed', fontsize=12)
ax[1].set_ylabel('Sum of relative differences of all the objectives', fontsize=12)
plt.show()

bestind = (nc,sn)

surr_revenue[bestind], surr_carbon[bestind], surr_deadwood[bestind], surr_ha[bestind]

(np.array((surr_revenue[bestind], surr_carbon[bestind], surr_deadwood[bestind], surr_ha[bestind])) - ide)/ide*100

real_revenue[bestind], real_carbon[bestind], real_deadwood[bestind], real_ha[bestind]

(np.array((real_revenue[bestind], real_carbon[bestind], real_deadwood[bestind], real_ha[bestind])) - ide)/ide*100

