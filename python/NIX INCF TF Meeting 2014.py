from __future__ import print_function
import nixio as nix
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

nf = nix.File.open('nix-demo-1.nix.h5', nix.FileMode.Overwrite)

step = 0.1
x = np.arange(0, 100, step)
y = np.sin(x)
print(x[:10])
print(y[:10])
plt.plot(x, y)

block = nf.create_block('test data set', 'nix.test.dataset')
print(block)

array = block.create_data_array('the_data', 'sin_data', nix.DataType.Double, y.shape)
array.data[:] = y

print(array.data[:10])

dim = array.create_sampled_dimension(1, step)

dim.label = 'x'
dim.unit = 's'

nf = nix.File.open('2014-08-21-am.nix.h5')

for b in nf.blocks:
    print(b.id, b.name)

block = nf.blocks[0]

for da in block.data_arrays:
    print(da.id, da.name, da.type)

for mt in block.multi_tags:
    print('* ', mt.name, mt.type, mt.id)
    print('-'*10)
    for ref in mt.references:
        print(ref.name, ref.id)
    print('')

for s in nf.sections:
    print("* %s: %s" % (s.name, s.type))
    for sub in s.sections:
        print("|- %s: %s" % (sub.name, sub.type))
        for subsub in sub.sections:
            print("| |-%s, %s [%d]" % (subsub.name, subsub.type, len(subsub)))
            for p in subsub.props:
                print("| | |~ %s: %s %s" % (p.name, p.values[0].value, p.unit or ""))
            print("| |")
        print("| ")

v1 = block.data_arrays['456a51b4-5d5d-4896-8e4b-e24c82c0eac2']

print(v1.data.shape)

d1 = v1.dimensions[0]
print (d1.dimension_type)

print(d1.label, d1.unit, d1.sampling_interval)

x_start = d1.offset or 0

howmany = 10000
x = np.arange(0, howmany) * d1.sampling_interval
y = v1.data[:howmany]
plt.plot(x, y)
plt.xlabel('%s [%s]' % (d1.label, d1.unit))
plt.ylabel('%s [%s]' % (v1.label, v1.unit))
plt.title('%s [%s]' % (v1.name, v1.type))
plt.xlim([np.min(x), np.max(x)])

from plotting import Plotter

pl = Plotter()
pl.xrange = np.s_[0:15]
pl.add_plot(block.data_arrays['b9425dc9-2d3b-4062-9ce7-eca6c2414510'])

pl = Plotter()
pl.add_plot(block.data_arrays['84cfeb28-c41a-4ff0-bd3f-d581b6250aa0'])

pl = Plotter()
pl.add_plot(block.data_arrays['f418663e-0fd2-46d9-9d28-cb2488b426a2'])

#File is created with gen-demo-data.py
nf = nix.File.open("demo.h5", nix.FileMode.ReadOnly)
print(nf.blocks)

block = nf.blocks[0]
mea = block.data_arrays['MEA']
print(mea.data.shape)
print("Sampling intervals: ", mea.dimensions[0].sampling_interval, mea.dimensions[0].unit, mea.dimensions[1].sampling_interval, mea.dimensions[1].unit)

pl = Plotter()
pl.add_plot(mea)

import cv2

nf = nix.File.open("2014-08-22_0.h5", nix.FileMode.ReadOnly)
block = nf.blocks[0]

for da in block.data_arrays:
    print("%10s\t%s \t [%s]" % (da.name, da.type, da.id))

video = block.data_arrays['5eeae436-db18-48e1-a7ed-2b73deb25014']

print(video.data.shape)

for i, d in enumerate(video.dimensions):
    print('%d: %s' % (i+1, d.dimension_type))
    if d.dimension_type == nix.DimensionType.Sample:
        print(' %s ' % d.label)
    if d.dimension_type == nix.DimensionType.Set:
        print(d.labels)
    if d.dimension_type == nix.DimensionType.Range:
        print("%s [%s]" % (d.label, d.unit))
        print(d.ticks)
    print('')

f1_r = video.data[:, :, 0, 0]

print(f1_r.shape)

plt.imshow(f1_r, cmap=mpl.cm.get_cmap('Greys'))

#not really working here in the IPython notebook
for k in range(0, video.data.shape[3]):
    img = video.data[:, :, :, k]
#    cv2.imshow('frame', img)

for t in block.multi_tags:
    print("%s %s [%s]"  % (t.name, t.type, t.id))

mt = block.multi_tags['76ebce5e-79aa-4886-af10-20e13bd76066']
pos, ext = mt.positions, mt.extents
print(pos.data.shape)
for k in range(pos.data.shape[1]):
    print('tag: %d' % k)
    print(' pos: %s' % str(pos.data[:, k]))
    print(' ext: %s' % str(ext.data[:, k]))
    print('')

f_pos_t = pos.data[3, 0]
f_pos_i = filter(lambda x: x[1] == f_pos_t, enumerate(video.dimensions[3].ticks))[0][0]
f_tagged = video.data[:, :, 1, f_pos_i]
plt.imshow(f_tagged)





