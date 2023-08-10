from IPython.display import Image
Image(filename='Simm_2014_fig5-62.jpg', width=800)

import numpy as np
import matplotlib.pyplot as plt
import bruges as b

get_ipython().magic('matplotlib inline')
# comment out the following if you're not on a Mac with HiDPI display
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

def get_elprop(ip,pr):
    vpvs= np.sqrt( (1-pr)/ (0.5-pr) )
    rho = (ip * 0.31**4)**(1/5)
    vp = ip / rho
    vs = vp / vpvs
    return vp,vs,rho

ip1=5500
pr1=0.4
vp1, vs1, rho1 = get_elprop(ip1,pr1)
print('First rock (green): given Ip={:.0f} and PR={:.2f} --> vp={:.0f}, vs={:.0f}, rho={:.2f}'.format(ip1,pr1,vp1,vs1,rho1))

ip2=5900
pr2=0.2
vp2, vs2, rho2 = get_elprop(ip2,pr2)
print('Second rock (yellow): given Ip={:.0f} and PR={:.2f} --> vp={:.0f}, vs={:.0f}, rho={:.2f}'.format(ip2,pr2,vp2,vs2,rho2))

def elastic_impedance(vp, vs, rho, alpha, k=0.25):
    alpha = np.radians(alpha)
    q = np.tan(alpha)**2
    w = -8*k * (np.sin(alpha)**2)
    e = 1 - 4*k * (np.sin(alpha)**2)
    rho_star = vp**q *  vs**w * rho**e
    ei = vp * rho_star
    return ei

def elastic_impedance_norm(vp, vs, rho, alpha, scal, k=0.25):
    alpha = np.radians(alpha)
    vp0, vs0, rho0 = scal[0], scal[1], scal[2]
    a = 1 + (np.tan(alpha)) ** 2
    b = -8 * k * ((np.sin(alpha)) ** 2)
    c = 1 - 4 * k * ((np.sin(alpha)) ** 2)
    ei = vp0*rho0 * ( (vp/vp0) ** a * (vs/vs0) ** b * (rho/rho0) ** c)
    return ei

n_samples = 400
interface1 = int(0.4*n_samples)
interface2 = int(0.6*n_samples)

model=np.zeros((n_samples,3))
model[:interface1,:]=[vp1,vs1,rho1]
model[interface2:,:]=[vp1,vs1,rho1]
model[interface1:interface2,:]=[vp2,vs2,rho2]

z=np.arange(n_samples)
ip=model[:,0]*model[:,2]
pr=(model[:,0]**2-2*model[:,1]**2) / (2*(model[:,0]**2-model[:,1]**2))

scal = vp1, vs1, rho1

ang=np.arange(0,50,10)

wavelet = b.filters.ricker(0.25, 0.001, 20)

ei  = np.zeros((n_samples,ang.size))
ei_norm  = np.zeros((n_samples,ang.size))
synt  = np.zeros((n_samples,ang.size))
for i,alpha in enumerate(ang):
    ei[:,i] = elastic_impedance(model[:,0], model[:,1], model[:,2], alpha)
    ei_norm[:,i] = elastic_impedance_norm(model[:,0], model[:,1], model[:,2], alpha, scal)
    RC = (ei[1:,i] - ei[:-1,i]) / (ei[1:,i] + ei[:-1,i])
    RC = np.append(np.nan, RC)
    RC = np.nan_to_num(RC)
    synt[:,i] = np.convolve(RC, wavelet, mode='same')

f=plt.subplots(figsize=(10, 8))
ax0 = plt.subplot2grid((1,7), (0,0), colspan=1)
ax1 = plt.subplot2grid((1,7), (0,1), colspan=1)
ax2 = plt.subplot2grid((1,7), (0,2), colspan=3)
ax3 = plt.subplot2grid((1,7), (0,5), colspan=2)

# track 1: AI
ax0.plot(ip, z, '-k', linewidth=2)
ax0.set_xlim(4000,7000)

# track 2: PR
ax1.plot(pr, z, '-k', linewidth=2)
ax1.set_xlim(0,0.5)
ax1.set_yticklabels([])

# track 3: seismogram
for i in range(ang.size):
    trace=synt[:,i]*3
    ax2.plot(trace+i,z,color='k', linewidth=1)
    ax2.fill_betweenx(z,trace+i,i, where=trace+i>i, facecolor='b', linewidth=0)
    ax2.fill_betweenx(z,trace+i,i, where=trace+i<i, facecolor='r', linewidth=0)
ax2.set_xlim(-0.9,synt.shape[1]-.1)
ax2.set_yticklabels([])
ax2.set_xticks((0, 1, 2, 3, 4))
ax2.set_xticklabels((0, 10, 20, 30, 40))
    
# track 4: EI logs
colr=['k','r','c','m','g']
for i in range(ang.size):
    ax3.plot(ei_norm[:,i], z, linewidth=2, label=ang[i], color=colr[i])
ax3.set_xlim(4000,7000)
ax3.set_yticklabels([])
ax3.legend(fontsize='small')

ax0.set_title('AI')
ax1.set_title('PR')
ax2.set_title('Gather')
ax3.set_title('EI')

for aa in [ax0, ax1, ax2, ax3]:
    aa.set_ylim(100,300)
    aa.invert_yaxis()
    aa.grid()
    plt.setp(aa.xaxis.get_majorticklabels(), rotation=90, fontsize=8)

f,ax=plt.subplots(nrows=1, ncols=2, figsize=(6, 8))
for i in range(ang.size):
    ax[0].plot(ei[:,i], z, linewidth=2, label=ang[i], color=colr[i])
    ax[1].plot(ei_norm[:,i], z, linewidth=2, label=ang[i], color=colr[i])

ax[0].set_title('EI')
ax[1].set_title('EI normalized')
ax[1].set_yticklabels([])
ax[1].set_xlim(4000,7000)

for aa in ax:
    aa.set_ylim(100,300)
    aa.invert_yaxis()
    aa.grid()
    plt.setp(aa.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
    aa.legend(fontsize='small', loc='upper right')

