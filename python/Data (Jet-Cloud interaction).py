import numpy as np
#import pyPLUTO as pp
from astropy.io import ascii
import os
import sys
from ipywidgets import interactive, widgets,fixed
from IPython.display import Audio, display
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation,FFMpegWriter
from matplotlib import rc,rcParams
from scipy.integrate import quad
rc('text', usetex=True)
rcParams['figure.figsize'] = (15., 6.0)
rcParams['ytick.labelsize'],rcParams['xtick.labelsize'] = 17.,17.
rcParams['axes.labelsize']=19.
rcParams['legend.fontsize']=17.
rcParams['text.latex.preamble'] = ['\\usepackage{siunitx}']
import seaborn
seaborn.despine()
seaborn.set_style('white', {'axes.linewidth': 0.5, 'axes.edgecolor':'black'})
seaborn.despine(left=True)
get_ipython().magic('load_ext autoreload')

get_ipython().magic('autoreload 1')

get_ipython().magic('aimport f')

def quadruple(d,VAR,tdk='Myrs',Save_Figure='',cl='',nn=0,mspeed='km',rows=2,cols=2,xlim=[None,None],
              ylim=[None,None],tlim=None,VARlim=[None,None],datafolder='../Document/DataImages/'):
    """
    Plot a rows(=2) x cols(=2) Variable 
    """
    X,Y=d['X'],d['Y']
    Vx=d['Vx'] if nn>0 else 0
    Vy=d['Vy'] if nn>0 else 0
    T=np.linspace(0,d['T'].shape[0]-1,rows*cols,dtype=int) if tlim==None else np.linspace(0,tlim,rows*cols,dtype=int)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True,
                            figsize=(cols*5,rows*5))
    i=0
    td=1e3 if tdk=='kyrs' else 1e6
    vmin=VARlim[0] if VARlim[0]==None else VAR.min()
    vmax=VARlim[1] if VARlim[1]==None else VAR.max()
    
    for ax in axes.flat:
        ext=[X.min(),X.max(),Y.min(),Y.max()]
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        #ax.add_artist(plt.Circle((0, 0), 1.0, color='r',fill=False,linestyle='--'))
        label = '{:.1f} {}'.format(d['T'][T[i]]/td,tdk)
        ax.set_title(label,fontsize=20)
        ax.grid(False)
        pc = ax.imshow(VAR[:,:,T[i]].T,cmap='viridis',origin='lower',aspect='equal',
                       extent=ext,vmin=vmin,vmax=vmax)
        if nn>0:
            k=nn #distance from boundaries for first/last arrows
            sc=2. if mspeed =='max' else 5. if mspeed == 'c' else 1e-4
            q=pc.axes.quiver(X[k:-k:nn],Y[k:-k:nn],
                            Vx[:,:,T[i]][k:-k:nn,k:-k:nn].T,
                            Vy[:,:,T[i]][k:-k:nn,k:-k:nn].T,
                             scale=sc,alpha=0.5,width=0.002)
            if mspeed == 'c':
                pc.axes.quiverkey(q,0.05,1.02,1.,r'$1\si{c}$',labelpos='E',fontproperties={'weight': 'bold'})
            elif mspeed == 'max':
                mV=np.max(np.sqrt(Vx[np.argmin((d['Y']-ylim[0])**2):np.argmin((d['Y']-ylim[1])**2),
                                     np.argmin((d['X']-xlim[0])**2):np.argmin((d['X']-xlim[1])**2),T[i]]**2+
                                  Vy[np.argmin((d['Y']-ylim[0])**2):np.argmin((d['Y']-ylim[1])**2),
                                     np.argmin((d['X']-xlim[0])**2):np.argmin((d['X']-xlim[1])**2),T[i]]**2))
                pc.axes.quiverkey(q,0.05,1.02,mV,'{:.2f} c'.format(mV),labelpos='E',
                                  fontproperties={'weight': 'bold'})
            else:
                pc.axes.quiverkey(q,0.02,1.02,3.36e-6,r'$1\si{km.s^{-1}}$',labelpos='E',fontproperties={'weight': 'bold'})
            
        i=i+1
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    plt.tight_layout()
    cbar_ax = fig.add_axes([0., 1.015, 1., 0.025*(np.float(cols)/rows)])#*(np.float(cols)/rows)
    cb=fig.colorbar(pc, cax=cbar_ax,orientation="horizontal",label=cl)
    cb.ax.tick_params(labelsize=17)
    cb.ax.xaxis.offsetText.set(size=20)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    form='.png'
    if Save_Figure <> '': plt.savefig(datafolder+Save_Figure+form,bbox_inches='tight',format='png', dpi=100)


#JC=np.load('../Data/jet-3c.npz')
#aJC=np.load('../Data/afterJet.npz')
hJC=np.load('../Data/jet-3c-n.npz')
sJC=np.load('../Data/jet-sneq.npz')
nJC=np.load('../Data/jet-3c-nn.npz')
tJC=np.load('../Data/jet-3c-tab.npz')
atJC=np.load('../Data/afterjet-tab.npz')

plt.figure(figsize=(10,10))
plt.imshow(np.log10(nJC['RHO'][:,:,0].T),origin='low',extent=[-150,150,-150,150])
Save_Figure='Jet0'
datafolder='../Document/DataImages/'
form='.png'
#plt.savefig(datafolder+Save_Figure+form,bbox_inches='tight',format='png', dpi=100)

f.quadruple(nJC,np.sqrt(nJC['Vx']**2+nJC['Vy']**2),tdk='kyrs',tlim=16,xlim=[-4,4],ylim=[-14,-8],Save_Figure='VnoCool')

f.quadruple(nJC,np.log10(nJC['RHO']),tdk='kyrs',tlim=16,xlim=[-4,4],ylim=[-14,-8],Save_Figure='RHOnoCool')

f.quadruple(nJC,np.log10(nJC['RHO']),tlim=200,tdk='kyrs')#,tlim=8,xlim=[-2,2],ylim=[-14,-12],VARlim=[0.,10.],Save_Figure='RHOnoCool0-10')

f.quadruple(nJC,np.sqrt(nJC['Vx']**2+nJC['Vy']**2),tdk='kyrs',tlim=8,xlim=[-2,2],ylim=[-14,-12])

f.quadruple(tJC,np.sqrt(tJC['Vx']**2+tJC['Vy']**2),tdk='kyrs',tlim=100)

f.quadruple(nJC,np.log10(nJC['RHO']),tdk='kyrs',tlim=100)

f.quadruple(tJC,np.log10(tJC['RHO']),tdk='kyrs',tlim=100)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

V=np.sqrt(tJC['Vx'][:,:,-1]**2+tJC['Vy'][:,:,-1]**2)

fV=np.fft.fft2(V)
fsV=np.abs(np.fft.fftshift(fV))
kx=np.fft.fftshift(np.fft.fftfreq(fsV.shape[0],10))
ky=np.fft.fftshift(np.fft.fftfreq(fsV.shape[1],10))

plt.imshow(np.log10(fsV**2),extent=[kx[0],kx[-1],ky[0],ky[-1]])

k=kx[256:]**2+ky[256:]**2
radprof=radial_profile(fsV[:256,:256]**2,[256,256])[:256]
plt.loglog(k,radprof)

from scipy.optimize import curve_fit

def fit(x,A,B): return A-B*x

y=np.log10(radprof[1:])
x=np.log10(k[1:])
s,sd2=curve_fit(fit,x,y,[1e8,5./3])

plt.loglog(k,radprof)
plt.loglog(k,10**s[0]*k**-s[1],label='{:.2f} Cascade'.format(-s[1]))
plt.legend()

mm=max(JC['T'].shape[0],aJC['T'].shape[0])
TT=np.arange(mm)
times=JC['T'] if JC['T'].shape[0]>aJC['T'].shape[0] else aJC['T']

def get_gas(RHO,aV,Temp,TT):
    #return np.array([RHO[:,:,t][np.logical_and(np.logical_and(aV[:,:,t]>6.7e-4,aV[:,:,t]<2.3e-3),Temp[:,:,t]<1e7)].sum()/RHO[:,:,t].sum() for t in TT])
    return np.array([RHO[:,:,t][aV[:,:,t]>6.7e-4].sum()/RHO[:,:,t].sum() for t in TT])
    #return np.array([RHO[:,:,t][aV[:,:,t]>2e-3].sum()/RHO[:,:,t].sum() for t in TT])

get_ipython().run_cell_magic('time', '', "Mftj7=get_gas(tJC['RHO'],np.sqrt(tJC['Vx']**2+tJC['Vy']**2),tJC['PRS']*f.Temp0/tJC['RHO'],np.arange(tJC['T'].shape[0]))\nMfnj7=get_gas(nJC['RHO'],np.sqrt(nJC['Vx']**2+nJC['Vy']**2),nJC['PRS']*f.Temp0/nJC['RHO'],np.arange(nJC['T'].shape[0]))")

ttlim=150
plt.plot(tJC['T'][:tlim]/1e3,Mftj[:ttlim],label='Escaped Gas ($V>\SI{600}{km.s^{-1}}$)')
#plt.plot(tJC['T'][:tlim]/1e3,Mfnj[:tlim],label='No Cooling')
plt.plot(tJC['T'][:tlim]/1e3,Mftj7[:ttlim],label='Gas in Wind ($V>\SI{200}{km.s^{-1}}$)')
#plt.plot(tJC['T'][:tlim]/1e3,Mfnj[:tlim],label='No Cooling')
plt.xlabel('Time (kyrs)')
plt.legend()
plt.savefig('/home/astromix/astro/MasterThesis/Document/DataImages/RatioEscapedGas.png',bbox_inches='tight')
f.quadruple(tJC,np.log10(tJC['RHO']),rows=3,tdk='kyrs',tlim=ttlim,Save_Figure='JetCloudRHO')
f.quadruple(tJC,np.sqrt(tJC['Vx']**2+tJC['Vy']**2),rows=3,tdk='kyrs',tlim=ttlim,Save_Figure='JetCloudV')

f.quadruple(tJC,np.sqrt(tJC['Vx']**2+tJC['Vy']**2),tdk='kyrs',tlim=8,xlim=[-2,2],ylim=[-14,-12],Save_Figure='JetISMV')
f.quadruple(tJC,np.log10(tJC['RHO']),tdk='kyrs',tlim=8,xlim=[-2,2],ylim=[-14,-12],Save_Figure='JetISMRHO')

f.quadruple(tJC,np.sqrt(tJC['Vx']**2+tJC['Vy']**2),tdk='kyrs',tlim=30,xlim=[-2,2],ylim=[-11,0])#,Save_Figure='JetISMV')
f.quadruple(tJC,np.log10(tJC['RHO']),tdk='kyrs',tlim=30,xlim=[-2,2],ylim=[-11,0])#,Save_Figure='JetISMRHO')

p=85
plt.plot(times[TT][0:min(mm,Mfj.shape[0])]/1e3,Mfj)
plt.plot(times[TT][p+1:min(mm,Mfaj.shape[0]+p)]/1e3,Mfaj[1:mm-p])

Temp=JC['PRS']*f.Temp0/JC['RHO']
Tempm=np.ma.masked_where(JC['RHO']<10,Temp)
plt.imshow(Tempm[:,:,-1].T,cmap='viridis',origin='lower')
#plt.contour(np.log10(JC['RHO'][:,:,-1]).T,origin='lower',levels=[1.,2.,3.])
plt.colorbar()

aTemp=aJC['PRS']*f.Temp0/aJC['RHO']
aTempm=np.ma.masked_where(aJC['RHO']<10,aTemp)
plt.imshow(aTempm[:,:,-1].T,cmap='viridis',origin='lower')
#plt.contour(np.log10(JC['RHO'][:,:,-1]).T,origin='lower',levels=[1.,2.,3.])
plt.colorbar()

plt.plot(times[TT][0:min(mm,Mfj.shape[0])]/1e3,Tempm.mean(axis=(0,1)))
plt.plot(times[TT][p+1:min(mm,Mfaj.shape[0]+p)]/1e3,aTempm.mean(axis=(0,1))[1:mm-p])

plt.plot(times[TT][0:min(mm,Mfj.shape[0])]/1e3,Temp.mean(axis=(0,1)))
plt.plot(times[TT][p+1:min(mm,Mfaj.shape[0]+p)]/1e3,aTemp.mean(axis=(0,1))[1:mm-p])
plt.yscale('log')

plt.plot(times[TT][0:min(mm,Mfj.shape[0])]/1e3,(np.sqrt(JC['Vx']**2+JC['Vy']**2)).mean(axis=(0,1)))
plt.plot(times[TT][p+1:min(mm,Mfaj.shape[0]+p)]/1e3,(np.sqrt(aJC['Vx']**2+aJC['Vy']**2)).mean(axis=(0,1))[1:mm-p])

plt.plot(times[TT][0:min(mm,Mfj.shape[0])]/1e3,(np.sqrt(JC['Vx']**2+JC['Vy']**2)).std(axis=(0,1)))
plt.plot(times[TT][p+1:min(mm,Mfaj.shape[0]+p)]/1e3,(np.sqrt(aJC['Vx']**2+aJC['Vy']**2)).std(axis=(0,1))[1:mm-p])
#plt.yscale('log')

get_ipython().run_cell_magic('time', '', "FF=tJC\nvfile='TabulatedV'#'test'\nstep=1\n\nVAR=np.sqrt(FF['Vx']**2+FF['Vy']**2)####\n#VAR=np.log10(FF['RHO'])\nT = FF['T'][::step]\nfig=plt.figure(figsize=(10,10))\nfig.set_tight_layout(True)\nax1=plt.subplot()\nax1.get_yaxis().get_major_formatter().set_useOffset(False)\next=[FF['X'].min(),FF['X'].max(),FF['Y'].min(),FF['Y'].max()]\n\npc = ax1.imshow(VAR[:,:,0].T,cmap='viridis',origin='lower',aspect='equal',extent=ext,\n                            vmin=VAR.min(axis=2).min(),vmax=VAR.max(axis=2).max())\ndef update(i):\n    ax1.cla()\n    pc = ax1.imshow(VAR[:,:,i].T,cmap='viridis',origin='lower',aspect='equal',extent=ext,\n                    vmin=VAR.min(axis=2).min(),vmax=VAR.max(axis=2).max())\n    label = 'Time = {0:.1f} kyrs'.format(T[i]/1000.)\n    ax1.set_title(label)\n    return ax1\nanim = FuncAnimation(fig, update, frames=range(T.shape[0]), interval=150)\nanim.save(vfile+'.gif',writer='imagemagic')")



