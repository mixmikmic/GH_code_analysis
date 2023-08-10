get_ipython().magic('matplotlib inline')

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

for g_ON in [100]:

    runfiles = np.load('fuse_threshold_data/FuseThreshold_L_x64_ON'+str(g_ON)+'Uniform01_run1.npz')
    event_list = runfiles['avalanches']
    voltages = runfiles['voltages']
    conductivities = runfiles['conductivities']
    # Recall conductivities is 1 longer than the others

    cond_4plot = []
    volt_4plot = []
    for n, v in enumerate(voltages):
        volt_4plot.extend([v, v])
        cond_4plot.extend([conductivities[n], conductivities[n+1]])
    
    fig, ax = plt.subplots(figsize=(11,7))

    ax.plot(volt_4plot, (np.array(cond_4plot) - 1)/(g_ON-1), lw=2, alpha=0.8)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)

    ax.set_title(r'$g_{on}='+str(g_ON)+'$', fontsize=24, y=1.01)
    ax.set_xlabel('Applied Voltage', fontsize=24)
    ax.set_ylabel('Network Conductance', fontsize=24)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


    #ax.set_yticks([10, 20, 30, 40])
    ax.set_xticks([20, 40, 60, 80])
    fig.savefig('Advancement_presentation/2DL45ON'+str(g_ON)+'_cond.png')

fig, ax = plt.subplots(figsize=(8.3,5))

for g_ON, c in zip([2, 20, 100, 1000], ['b', 'g', 'r', 'k']) :

    try:
        runfiles = np.load('fuse_threshold_data/FuseThreshold_L_x64_ON'+str(g_ON)+'Uniform01_run1.npz')
    except:
        runfiles = np.load('fuse_threshold_data/FuseThreshold_L_x64_ON'+str(g_ON)+'Uniform01_run2.npz')
        
    event_list = runfiles['avalanches']
    voltages = runfiles['voltages']
    conductivities = runfiles['conductivities']
    # Recall conductivities is 1 longer than the others

    cond_4plot = []
    volt_4plot = []
    for n, v in enumerate(voltages):
        volt_4plot.extend([v, v])
        cond_4plot.extend([conductivities[n], conductivities[n+1]])


    ax.plot(volt_4plot, (np.array(cond_4plot) - 1)/(g_ON-1), lw=2, alpha=0.7, c=c)
    
ax.set_xlim(0, 40)
ax.set_ylim(0, 0.6)

#ax.set_title(r'$g_{on}='+str(g_ON)+'$', fontsize=24, y=1.01)
ax.set_xlabel('Applied Voltage (V)', fontsize=20)
ax.set_ylabel(r'Network Conductance ($\frac{G-G_{off}}{G_{on}-G_{off}}$)', fontsize=20)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.text(28.5, 0.16, r'$G_{on}=10$',fontdict=dict(fontsize=17, color='b', alpha=0.7))
ax.text(28.5, 0.295, r'$G_{on}=20$',fontdict=dict(fontsize=17, color='g', alpha=0.7))
ax.text(28.5, 0.396, r'$G_{on}=100$',fontdict=dict(fontsize=15, color='r', alpha=0.7))
ax.text(28, 0.54, r'$G_{on}=1000$',fontdict=dict(fontsize=17, color='k', alpha=0.7))

    #ax.set_yticks([10, 20, 30, 40])
ax.set_xticks([10, 20, 30, 40])
fig.tight_layout()
fig.savefig('MF_paper/PT_Networks_Conductances.png')

fig, ax = plt.subplots(figsize=(8.3,5))

for g_ON, c in zip([2, 20, 100, 1000], ['b', 'g', 'r', 'k']) :

    try:
        runfiles = np.load('fuse_threshold_data/FuseThreshold_L_x64_ON'+str(g_ON)+'Uniform01_run1.npz')
    except:
        runfiles = np.load('fuse_threshold_data/FuseThreshold_L_x64_ON'+str(g_ON)+'Uniform01_run2.npz')
        
    event_list = runfiles['avalanches']
    voltages = runfiles['voltages']
    conductivities = runfiles['conductivities']
    # Recall conductivities is 1 longer than the others

    cond_4plot = []
    volt_4plot = []
    for n, v in enumerate(voltages):
        volt_4plot.extend([v, v])
        cond_4plot.extend([conductivities[n], conductivities[n+1]])


    ax.plot(volt_4plot, (np.array(cond_4plot) - 1)/(g_ON-1), lw=2, alpha=0.7, c=c)
    
ax.set_xlim(0, 40)
ax.set_ylim(0, 0.6)

#ax.set_title(r'$g_{on}='+str(g_ON)+'$', fontsize=24, y=1.01)
ax.set_xlabel('Applied Voltage (V)', fontsize=20)
ax.set_ylabel(r'Network Conductance ($\frac{G-G_{off}}{G_{on}-G_{off}}$)', fontsize=20)

#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.text(28.5, 0.16, r'$G_{on}=10$',fontdict=dict(fontsize=17, color='b', alpha=0.7))
ax.text(28.5, 0.295, r'$G_{on}=20$',fontdict=dict(fontsize=17, color='g', alpha=0.7))
ax.text(28.5, 0.396, r'$G_{on}=100$',fontdict=dict(fontsize=15, color='r', alpha=0.7))
ax.text(28, 0.54, r'$G_{on}=1000$',fontdict=dict(fontsize=17, color='k', alpha=0.7))

im = plt.imread('Advancement_presentation/Inside_a_network_3.png')
inset=plt.axes([0.12, 0.45, .32, .435])
inset.imshow(im)
inset.axis('off')

ax.text(3, 0.57, 'V+', fontdict=dict(fontsize=12, alpha=0.7))

    #ax.set_yticks([10, 20, 30, 40])
ax.set_xticks([10, 20, 30, 40])
fig.tight_layout()
fig.savefig('MF_paper/PT_Networks_Conductances-2.png')

fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(voltages, [sum(i) for i in event_list])

fig, ax = plt.subplots(figsize=(15,8))
#fig.subplots_adjust(bottom=0.3)

ax.plot(voltages, [sum(i) for i in event_list])
ax.set_ylabel('Avalanche Size', fontsize=24)
ax.set_xlabel('Voltage', fontsize=24)
ax.set_yticks([0, 200, 400, 600, 800])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

fig.savefig('Advancement_presentation/Avalanche_fullrun.png')

fig, ax = plt.subplots(figsize=(15,8))
#fig.subplots_adjust(bottom=0.3)

ax.plot(voltages, [sum(i) for i in event_list])
ax.set_ylabel('Avalanche Size', fontsize=24)
ax.set_xlabel('Voltage', fontsize=24)
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 200)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

fig.savefig('Advancement_presentation/Avalanche_beginrun.png')

fig, ax = plt.subplots(figsize=(15,8))
#fig.subplots_adjust(bottom=0.3)

ax.plot(voltages, [sum(i) for i in event_list])
ax.set_ylabel('Avalanche Size', fontsize=24)
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 200)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks([])

inset=plt.axes([0.2, 0.45, .25, .4])

inset.yaxis.set_ticks_position('left')
inset.xaxis.set_ticks_position('bottom')
inset.set_xticks([])
inset.set_yticks([])

g_ON = 100
g_OFF = 1

def G_effectivemedium(f):
    return ((2*f-1) * (g_ON-g_OFF) + np.sqrt((2*f-1)**2 * (g_ON-g_OFF)**2 + 4*g_ON*g_OFF)) / 2

def f_of_phi(phi):
    return (phi - g_OFF) / (g_ON - g_OFF)

def phi_of_f(f):
    return g_OFF + (g_ON - g_OFF) * f

diff2sub=0.27
sub2crit=0.45

f_diff = np.linspace(0, diff2sub, 100)
f_sub = np.linspace(diff2sub,sub2crit,100)
f_sup= np.linspace(sub2crit, 1, 100)
inset.plot(f_diff, np.sqrt(G_effectivemedium(f_diff) / phi_of_f(f_diff)), color='b', alpha=0.7, lw=2)

inset.set_ylim(0.25, 1)
inset.set_xlim(0, 1)
inset.set_ylabel(r'$h(f)$', fontsize=20)
inset.set_xlabel(r'$f$', fontsize=20)




ax.annotate('', xy=(0,0), xycoords='data',
                         xytext=(5.,0), textcoords='data',
                         arrowprops=dict(arrowstyle='<|-|>',
                                         connectionstyle='bar,fraction=-0.07'))
ax.text(1.8, -20, 'Diffusive', fontdict=dict(fontsize=16, color='b', alpha=0.7))
fig.savefig('Advancement_presentation/Diffusive.png')

ax.annotate('', xy=(5.1,0), xycoords='data',
                         xytext=(11.5,0), textcoords='data',
                         arrowprops=dict(arrowstyle='<|-|>',
                                         connectionstyle='bar,fraction=-0.054'))
ax.text(6.7, -20, 'Subcritical Branching', fontdict=dict(fontsize=16, color='darkorange'))
inset.plot(f_sub, np.sqrt(G_effectivemedium(f_sub) / phi_of_f(f_sub)), color='darkorange', lw=2)
fig.savefig('Advancement_presentation/SubCritical.png')

ax.annotate('', xy=(11.6,0), xycoords='data',
                         xytext=(12.5,-10.5), textcoords='data',
                         arrowprops=dict(arrowstyle='<|-|>',
                                         connectionstyle='angle,angleA=180,angleB=90,rad=0'))
ax.text(11.9, -20, 'Supercritical', fontdict=dict(fontsize=16, color='r'))
ax.text(11.9, -28, 'Branching', fontdict=dict(fontsize=16, color='r'))
inset.plot(f_sup, np.sqrt(G_effectivemedium(f_sup) / phi_of_f(f_sup)), color='r', lw=2)
fig.savefig('Advancement_presentation/SuperCrit.png')



limits = np.linspace(0, 12, num=31)
avalanches = np.array([sum(i) for i in event_list], dtype=int)
N_samples = np.zeros((30,))
Avalanche_sums = np.zeros((30,))

for i, lim in enumerate(limits[:-1]):
    interval_mask = np.logical_and(voltages>= limits[i], voltages<=limits[i+1])
    N_samples[i] += np.sum(interval_mask)
    Avalanche_sums[i] +=np.sum(avalanches[interval_mask])

fig, ax = plt.subplots()

ax.plot((limits[:-1] + limits[1:]) / 2., np.divide(Avalanche_sums, N_samples))



g_ON = 100
g_OFF = 1

def G_eff(f):
    return ((2*f-1) * (g_ON-g_OFF) + np.sqrt((2*f-1)**2 * (g_ON-g_OFF)**2 + 4*g_ON*g_OFF)) / 2

def f_of_phi(phi):
    return (phi - g_OFF) / (g_ON - g_OFF)

def phi(f):
    return g_OFF + (g_ON - g_OFF) * f

def deriv_phi(f):
    return g_ON - g_OFF

def deriv_G_eff(f):
    return (g_ON - g_OFF) + (g_ON-g_OFF)**2 * (2*f-1) / (np.sqrt((2*f-1)**2 * (g_ON - g_OFF)**2 + 4 * g_ON * g_OFF))

def h(f):
    return np.sqrt(G_eff(f) / phi(f))

def deriv_h(f):
    return 1 / (2 * h(f)) * (deriv_G_eff(f) * phi(f) - deriv_phi(f) * G_eff(f)) / phi(f)**2

def mu(f, v):
    return deriv_h(f) * v
    
def mean_avalanche(f, v):
    if mu(f, v) < 0:
        return 1
    elif mu(f, v) < 1:
        return 1 / (1 - mu(f, v))
    else:
        return 0
    
mean_avalanche_vec = np.vectorize(mean_avalanche)

fig, ax = plt.subplots(1, 4, figsize=(15,5))

f = np.linspace(0, 1)
ax[0].plot(f, phi(f))
ax[0].plot(f, G_eff(f))
ax[1].plot(f, G_eff(f))
ax[1].plot(f, deriv_G_eff(f))
ax[2].plot(f, h(f))
ax[2].plot(f, deriv_h(f))
ax[2].set_ylim(-1, 3)
ax[3].plot(f, mu(f, ))

limits = np.linspace(0, 12, num=31)
N_samples = np.zeros((30,))
Avalanche_sums = np.zeros((30,))
Avalanche_square_sums = np.zeros((30,))
f_vec = np.zeros((30,))


for run in range(1,500):
    filename = "fuse_threshold_data/FuseThreshold_L_x64_ON100Uniform01_run" + str(run) + ".npz"
    runfiles = np.load(filename)
    avalanches = np.array([sum(i) for i in runfiles['avalanches']], dtype=int)
    voltages = runfiles['voltages']
    f = np.cumsum(avalanches)
    f = f / f[-1]
    for i, lim in enumerate(limits[:-1]):
        interval_mask = np.logical_and(voltages>= limits[i], voltages<=limits[i+1])
        N_samples[i] += np.sum(interval_mask)
        Avalanche_sums[i] +=np.sum(avalanches[interval_mask])
        f_vec[i] += np.sum(f[interval_mask])
        
f_avg = np.divide(f_vec, N_samples)

fig, ax = plt.subplots(figsize=(10, 5))

v_avg = (limits[:-1] + limits[1:]) / 2.
avalanche_avg = np.divide(Avalanche_sums, N_samples)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel('Average Avalanche Size', fontsize=20)
ax.set_xlabel('Voltage', fontsize=20)


ax.scatter(v_avg, avalanche_avg, c='b', alpha=0.7, edgecolors='none')
ax.set_ylim(0, 50)
ax.set_xlim(0, 12)
fig.savefig('Advancement_presentation/First_moment.png')

for g_ON in [2, 10, 100, 1000]:
    runfiles = np.load('fuse_threshold_data/FuseThreshold_L128_ON' + str(g_ON) + 'run1.npz')
    event_list = runfiles['avalanches']
    print len(event_list)



