get_ipython().run_line_magic('matplotlib', 'inline')
# plots graphs within the notebook
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg' # not sure what this does, may be default images to svg format")

from IPython.display import display,Image, Latex
from __future__ import division
from sympy.interactive import printing
printing.init_printing(use_latex='mathjax')
from IPython.display import clear_output

import time

from IPython.display import display,Image, Latex

from IPython.display import clear_output


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.constants as sc
import h5py

import sympy as sym

    
font = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }
fontlabel = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }

from matplotlib.ticker import FormatStrFormatter
plt.rc('font', **font)

class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
    
font = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }
fontlabel = {'family' : 'serif',
        #'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }

from matplotlib.ticker import FormatStrFormatter
plt.rc('font', **font)

t, u, tmp = np.genfromtxt('Data/p20_20.ts', delimiter='', unpack=True, dtype=float)
N = t.shape[0]
print('Number of samples: %6i'%N)
plt.plot(t[0:N+1:100],u[0:N+1:100],label = r"signal")
um_plot = np.mean(u)*np.ones(2)
t_line_plot = np.array([0, 90.])
plt.plot(t_line_plot,um_plot,'--',lw=2, label = r"mean")
plt.legend(loc=3, bbox_to_anchor=[0, 1],
          ncol=3, shadow=False, fancybox=True)
plt.xlabel('$t$ ($s$)', fontdict = fontlabel)
plt.ylabel('$u$ ($m/s$)', fontdict = fontlabel)
plt.show()

import matplotlib.mlab as mlab
n, bins, patches = plt.hist(u, 100, normed=1, facecolor='blue', alpha=0.75)
y = mlab.normpdf( bins, np.mean(u), np.std(u))
plt.plot(bins, y, 'r-', linewidth=2)

np.min(t[1:]-t[0:-1])
np.max(t[1:]-t[0:-1])
# plt.xlim(0,20)
# plt.ylim(1e-6,1e-5)
    

data = np.genfromtxt("Data/Nu.01.001.dat")
plt.plot(data[:,0],data[:,1])
plt.xlabel(r"$t$", fontdict=fontlabel)
plt.ylabel(r"$Nu$", fontdict=fontlabel)
plt.title(r"$Ra=10^5,Pr=4$")
plt.savefig("Nu_signal_2per.png", bbox_inches='tight')
plt.show()
print(np.mean(data[:,1]))

dt = 1e-3
fs = 1/dt
import scipy.signal as sig
Nu = np.copy(data[:,1])
f, p = sig.periodogram(Nu,fs)

plt.loglog(f,p)
plt.ylim(1e-9,1e2)
plt.xlabel(r"$f$", fontdict=fontlabel)
plt.ylabel(r"$PSD(Nu)$", fontdict=fontlabel)
titlestr = "$f_{dom}=$"
titlestr += "{0:.3f}".format(f[np.argmax(p)])
plt.title(titlestr)
plt.savefig("Nu_PSD.png", bbox_inches='tight')
plt.show()

from scipy.interpolate import interp1d
f = interp1d(t,u)
treg = np.linspace(t[0],t[-1],100000)

ureg = f(treg)

plt.plot(treg[::10],ureg[::10],lw=2, label = r"mean")
# plt.legend(loc=3, bbox_to_anchor=[0, 1],
#           ncol=3, shadow=False, fancybox=True)
plt.xlabel('$t$ ($s$)', fontdict = fontlabel)
plt.ylabel('$u$ ($m/s$)', fontdict = fontlabel)
plt.show()

dt = t[1] - t[0]
fs = 1/dt
import scipy.signal as sig
f, p = sig.periodogram(ureg,fs,nfft = 10000)

plt.loglog(f,p)
plt.ylim(1e-7,1e0)
# plt.xlim(1e-2,1e2)
plt.xlabel(r"$f$", fontdict=fontlabel)
plt.ylabel(r"$PSD(Nu)$", fontdict=fontlabel)
# titlestr = "$f_{dom}=$"
# titlestr += "{0:.3f}".format(f[np.argmax(p)])
# plt.title(titlestr)
# plt.savefig("Nu_PSD.png", bbox_inches='tight')
plt.show()
get_ipython().run_line_magic('pinfo', 'sig.periodogram')

class Profiles(object):
    def __init__(self,Rename):
        self.name = Rename
        
    def get_data(self,linetype):
        filename = 'Channel-data/Re'+self.name+'/profiles/Re'+self.name+'.prof'
        data_matrix = np.genfromtxt(filename, skip_header = 28, delimiter='', unpack=True, dtype=float)
        self.linetype = linetype
        self.N = len(data_matrix[0,:])
        self.Re = data_matrix[1,self.N-1]
        self.yoverh = data_matrix[0,:]
        self.yplus = data_matrix[1,:]
        self.uplus = data_matrix[2,:]
        self.urmsplus = data_matrix[3,:]
        self.vrmsplus = data_matrix[4,:]
        self.wrmsplus = data_matrix[5,:]
        self.dudyplus = data_matrix[6,:]
        self.omxrmsplus = data_matrix[7,:]
        self.omyrmsplus = data_matrix[8,:]
        self.omzrmsplus = data_matrix[9,:]
        self.uvplus = data_matrix[10,:]
        self.uwplus = data_matrix[11,:]
        self.vwplus = data_matrix[12,:]
        

hplusstr = np.array(['180', '550', '950', '2000'])
linetypestr = np.array(['b-', 'g-', 'c-', 'r-'])
profilesRe = []
i = 0
for name in hplusstr:
    profilesRe.append(Profiles(name))
    profilesRe[i].get_data(linetypestr[i])
    plt.plot(profilesRe[i].yplus,profilesRe[i].uplus,              profilesRe[i].linetype, lw=2, label=r'$Re_\tau=$ '+profilesRe[i].name)
    i += 1

plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=2, shadow=False, fancybox=True)
plt.xlabel(r'$y^+$', fontdict = font)
plt.ylabel(r'$\overline{u}^+$', fontdict = font)
plt.show()
    

nut = -profilesRe[0].uvplus[:-1]/profilesRe[0].dudyplus[:-1]

i = 0
for name in hplusstr:
    plt.semilogx(profilesRe[i].yplus,profilesRe[i].uplus,                  profilesRe[i].linetype, lw=2, label=r'$Re_\tau=$ '+profilesRe[i].name)
    i += 1
y_log = np.logspace(1, 3, 100)
u_log = (1./0.41)*np.log(y_log)+5.2
plt.semilogx(y_log,u_log,'k--', lw =2, label = r'$\overline{u}^+=\frac{1}{0.41}\ln(y^+)+5.2$')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=2, shadow=False, fancybox=True)
plt.xlabel(r'$y^+$', fontdict = font)
plt.ylabel(r'$\overline{u}^+$', fontdict = font)
plt.show()


i = 0
for name in hplusstr:
    plt.semilogx(profilesRe[i].yplus,profilesRe[i].dudyplus*profilesRe[i].yplus,                  profilesRe[i].linetype, lw=2, label=r'$Re_\tau=$ '+profilesRe[i].name)
    i += 1
IndFunc_log = (1./0.41)*np.ones(100)
plt.semilogx(y_log,IndFunc_log,'k--', lw =2, label = r'$\Phi=\frac{1}{0.41}$')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=2, shadow=False, fancybox=True)
plt.xlabel(r'$y^+$', fontdict = font)
plt.ylabel(r'$\Phi=y^+(d\overline{u}^+/dy^+)$', fontdict = font)
plt.show()


i = 0
for name in hplusstr:
    plt.semilogx(profilesRe[i].yplus,profilesRe[i].uplus/profilesRe[i].yplus,                  profilesRe[i].linetype, lw=2, label=r'$Re_\tau=$ '+profilesRe[i].name)
    i += 1

plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=2, shadow=False, fancybox=True)
plt.xlabel(r'$y^+$', fontdict = font)
plt.ylabel(r'$\overline{u}^+/y^+$', fontdict = font)
plt.grid(True)
plt.show()


i = 0
for name in hplusstr:
    plt.loglog(profilesRe[i].yplus[:-1],-profilesRe[i].uvplus[:-1]/profilesRe[i].dudyplus[:-1]/profilesRe[i].Re,                  profilesRe[i].linetype, lw=2, label=r'$Re_\tau=$ '+profilesRe[i].name)
    i += 1

plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=2, shadow=False, fancybox=True)
plt.xlabel(r'$y^+$', fontdict = font)
plt.ylabel(r'$\nu_T^+$', fontdict = font)
plt.grid(True)
plt.show()

class Statistics(object):
    def __init__(self):
        self.Re = 5185.897
        
#     def get_data(self):
        filename = 'Channel-data/Re5200/LM_Channel_5200_mean_prof.dat'
        data_matrix = np.genfromtxt(filename, skip_header = 73, delimiter='', unpack=True, dtype=float)
        self.N = len(data_matrix[0,:])
        self.Re = 5185.897
        self.yoverh = data_matrix[0,:]
        self.yplus = data_matrix[1,:]
        self.u = data_matrix[2,:]
        self.dudy = data_matrix[3,:]
        filename = 'Channel-data/Re5200/LM_Channel_5200_vel_fluc_prof.dat'
        data_matrix = np.genfromtxt(filename, skip_header = 76, delimiter='', unpack=True, dtype=float)
        self.uu = data_matrix[2,:]
        self.vv = data_matrix[3,:]
        self.ww = data_matrix[4,:]
        self.uv = data_matrix[5,:]
        self.k = data_matrix[8,:]
        filename = 'Channel-data/Re5200/LM_Channel_5200_RSTE_k_prof.dat'
        data_matrix = np.genfromtxt(filename, skip_header = 75, delimiter='', unpack=True, dtype=float)
        self.production = data_matrix[2,:]
        self.tdiff = data_matrix[3,:]
        self.vdiff = data_matrix[4,:]
        self.pstrain = data_matrix[5,:]
        self.pdiff = data_matrix[6,:]
        self.dissipation = data_matrix[7,:]
        

Sim5200 = Statistics()

nu_t_k_e = 0.09*np.power(Sim5200.k,2)/Sim5200.dissipation

plt.loglog(Sim5200.yplus,nu_t_k_e)
plt.loglog(Sim5200.yplus,-Sim5200.uv/Sim5200.dudy)

Ny = 200
Nz = 50
X = 40.*np.ones((Ny,Nz))
Y = np.linspace(-10,10,Ny)
Z = np.linspace(-2,2,Nz)
P = np.random.normal(size = (Ny,Nz),loc=0.0,scale=0.1)

print(P.shape,Y.shape,Z.shape)

import csv
with open('pressure.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for k in range(Nz):
        for j in range(Ny):
            filewriter.writerow([Y[j], Z[k], P[j,k]])

t = np.linspace(0,40,1000)
p = np.random.normal(size = 1000, loc=0.0, scale=0.1)
import csv
with open('tpressure.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for i in range(1000):
        filewriter.writerow([t[i], p[i]])

data = np.genfromtxt("ProfilesxoverD_2.csv", skip_header=1, delimiter =",")

print(data.shape)

plt.plot(data[:,16],data[:,0])

print(np.argmax(data[:,0]))

print(data[np.argmax(data[:,0]),16])



