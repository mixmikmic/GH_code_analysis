get_ipython().magic('pylab inline')

import sys, os
from ptha_paths import data_dir, events_dir

fixed_grid_file = os.path.join(data_dir, 'MapsTopo', 'fixedgrid_xyB_small.npy')
d=load(fixed_grid_file)
x=d[:,0]
y=d[:,1]
B=d[:,2]

print "x,y,B are 1-dimensional arrays of length %s" % len(x)

# to reshape them as 2-dimensional arrays:

topo = reshape(B, (250,250), order='F')
X = reshape(x, (250,250), order='F')
Y = reshape(y, (250,250), order='F')

print "X,Y,topo are 2-dimensional arrays with shape ", X.shape

fname = 'xyB.txt'
f = open(fname,'w')
for npts in range(len(B)):
    f.write('%24.10f %24.10f %24.10f\n' % (x[npts],y[npts],B[npts]))
f.close()
print "Created file ", os.path.abspath(fname)

fname = 'xyB.txt'
data_array = vstack((x, y, B)).T
savetxt(fname, data_array, fmt='%24.10f')
print "Created file ", os.path.abspath(fname)

event = 'AASZb'
event_dir = os.path.join(events_dir, event)
zeta_file = os.path.join(event_dir, 'h_eta_small.npy')
zeta = load(zeta_file)

fname = '%s.txt' % event
data_array = vstack((x, y, zeta)).T
savetxt(fname, data_array, fmt='%24.10f')
print "Created file ", os.path.abspath(fname)



