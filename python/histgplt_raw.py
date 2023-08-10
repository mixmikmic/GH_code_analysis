#general imports
import matplotlib.pyplot as plt   
import pygslib
import numpy as np

#make the plots inline
get_ipython().magic('matplotlib inline')

#get the data in gslib format into a pandas Dataframe
mydata= pygslib.gslib.read_gslib_file('../datasets/cluster.dat')  

# This is a 2D file, in this GSLIB version we require 3D data and drillhole name or domain code
# so, we are adding constant elevation = 0 and a dummy BHID = 1 
mydata['Zlocation']=0
mydata['bhid']=1

# printing to verify results
print ' \n **** 5 first rows in my datafile \n\n  ', mydata.head(n=5)

#view data in a 2D projection
plt.scatter(mydata['Xlocation'],mydata['Ylocation'], c=mydata['Primary'])
plt.colorbar()
plt.grid(True)
plt.show()

print pygslib.gslib.__plot.histplt.__doc__

mydata['Declustering Weight'].sum()

parameters_histplot = {
        'hmin' : 0.06,                          #in/output rank-0 array(float,'d')
        'hmax' : 20.06,                         #in/output rank-0 array(float,'d')
        'ncl'  : 40,                            #int, number of bins
        'iwt'  : 0,                             #int, 1 use declustering weight
        'ilog' : 1,                             #int, 1 use logscale
        'icum' : 0,                             #int, 1 use cumulative
        'va'   : mydata['Primary'],             # array('d') with bounds (nd)
        'wt'   : mydata['Declustering Weight']} # array('d') with bounds (nd), wight variable (obtained with declust?)


parameters_histplotd = {
        'hmin' : 0.06,                          #in/output rank-0 array(float,'d')
        'hmax' : 20.06,                         #in/output rank-0 array(float,'d')
        'ncl'  : 40,                            #int, number of bins
        'iwt'  : 1,                             #int, 1 use declustering weight
        'ilog' : 1,                             #int, 1 use logscale
        'icum' : 0,                             #int, 1 use cumulative
        'va'   : mydata['Primary'],             # array('d') with bounds (nd)
        'wt'   : mydata['Declustering Weight']} # array('d') with bounds (nd), wight variable (obtained with declust?)


binval,nincls,cl, clwidth,xpt025,xlqt,xmed,xuqt,xpt975, xmin,xmax,xcvr,xmen,xvar,xfrmx,dcl,error = pygslib.gslib.__plot.histplt(**parameters_histplot)

binvald,ninclsd,cld, clwidthd, xpt025d,xlqtd,xmedd,xuqtd, xpt975d,xmind,xmaxd,xcvrd,xmend,xvard,xfrmxd,dcld,errord = pygslib.gslib.__plot.histplt(**parameters_histplotd)

print dcl
print cl.round(1)
print nincls
print binval.round(2)
print clwidth
mydata.Primary[mydata.Primary>20.1]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Bin probability')
plt.bar (cl, binval, width=-clwidth, label = 'Non-declustered')
plt.bar (cld, binvald, width=-clwidth, alpha=0.5, color='r', label = 'Declustered')
if parameters_histplot['ilog']>0:
    ax.set_xscale('log')
plt.grid(True)
plt.legend(loc=2)
fig.show

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Bin count (Warning: this will not show the effect of weight)')
plt.bar (cl, nincls, width=-clwidth,label = 'Non-Declustered')
plt.bar (cld, ninclsd, width=-clwidth, alpha=0.5, color='r',label = 'Declustered')
if parameters_histplot['ilog']>0:
    ax.set_xscale('log')
plt.grid(True)
plt.legend(loc=2)
fig.show

parameters_histplot = {
        'hmin' : 0.06,                          #in/output rank-0 array(float,'d')
        'hmax' : 20.06,                         #in/output rank-0 array(float,'d')
        'ncl'  : 40,                            #int, number of bins
        'iwt'  : 0,                             #int, 1 use declustering weight
        'ilog' : 1,                             #int, 1 use logscale
        'icum' : 1,                             #int, 1 use cumulative
        'va'   : mydata['Primary'],             # array('d') with bounds (nd)
        'wt'   : mydata['Declustering Weight']} # array('d') with bounds (nd), wight variable (obtained with declust?)


parameters_histplotd = {
        'hmin' : 0.06,                          #in/output rank-0 array(float,'d')
        'hmax' : 20.06,                         #in/output rank-0 array(float,'d')
        'ncl'  : 40,                            #int, number of bins
        'iwt'  : 1,                             #int, 1 use declustering weight
        'ilog' : 1,                             #int, 1 use logscale
        'icum' : 1,                             #int, 1 use cumulative
        'va'   : mydata['Primary'],             # array('d') with bounds (nd)
        'wt'   : mydata['Declustering Weight']} # array('d') with bounds (nd), wight variable (obtained with declust?)


binval,nincls,cl, clwidth,xpt025,xlqt,xmed,xuqt,xpt975,xmin, xmax,xcvr,xmen,xvar,xfrmx,dcl,error = pygslib.gslib.__plot.histplt(**parameters_histplot)

binvald,ninclsd,cld, clwidthd,xpt025d,xlqtd,xmedd,xuqtd,xpt975d, xmind,xmaxd,xcvrd,xmend,xvard,xfrmxd,dcld,errord = pygslib.gslib.__plot.histplt(**parameters_histplotd)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Bin probability, bin style')
plt.bar (cld, binvald, width=-clwidth, color='r', label = 'Declustered')
plt.bar (cl, binval, width=-clwidth, label = 'Non-declustered')

if parameters_histplot['ilog']>0:
    ax.set_xscale('log')
plt.grid(True)
plt.legend(loc=2)
fig.show

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Bin probability, step style')
plt.step (cld, binvald, where='post', color='r', label = 'Declustered')
plt.step (cl, binval, where='post', label = 'Non-declustered')

if parameters_histplot['ilog']>0:
    ax.set_xscale('log')
plt.grid(True)
plt.legend(loc=2)
fig.show

print 'data min, max: ', xmin, xmax
print 'data quantile 2.5%, 25%, 50%, 75%, 97.75%: ' , xpt025,xlqt,xmed,xuqt,xpt975
print 'data cv, mean, variance : ',  xcvr,xmen,xvar
print 'hitogram max frequency (useful to rescal some plots)' , xfrmx
print 'error <> 0? Then all ok?' , error==0



