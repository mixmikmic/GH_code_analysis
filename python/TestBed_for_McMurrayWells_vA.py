import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import welly
welly.__version__

import os
env = get_ipython().magic('env')

well_path = 'SPE_006_originalData/OilSandsDB/Logs/'

from welly import Well

w = Well.from_las(well_path+'00-01-01-095-19W4-0.LAS')

w

tracks = ['CALI', 'GR', 'DPHI', 'NPHI', 'ILD']
w.plot(tracks=tracks)

r = Well.from_las(well_path+'00-01-04-075-23W4-0.LAS')
r

tracks = ['CALI', 'DPHI', 'GR', 'ILD', 'NPHI']
r.plot(tracks=tracks)

import lasio
l = lasio.read(well_path+"00-01-04-075-23W4-0.LAS")
l

l.curves

l.keys()

l_depth = l['DEPT']
l_depth



from welly import Project

p = Project.from_las('Logs/*.LAS')

p

keys = ['ILD','DPHI','GR','NPHI','CALI','COND','DELT','RHOB','PHIN','DT','ILM','SP','SFLU','IL','DEPTH','DEPH','MD']

keys2 = ['ILD','DPHI','GR','NPHI','CALI','RHOB']

html = p[1200:2000].curve_table_html(keys=keys)

from IPython.display import HTML
HTML(html)









