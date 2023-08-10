path = '../../../SEER/services/deepner/server'
import os
r = [ f for f in os.listdir(path) if f.endswith('.log') ]
r

import re
pattern = re.compile("\s+(\S+)\:\s+(\S+)")

def parse_log(filename):
    with open(filename, 'r') as f:
        d=dict()
        for l in f.readlines():
            if '--------------------------------------------' in l:
                if len(d)>0:
                    yield d 
                d=dict() # Empty out the data
            else:
                m = pattern.search(l)
                if m:
                    d[m.group(1)] = m.group(2)
        yield d

data = parse_log(os.path.join(path, 'train.10epochs-8.cpu.log'))

next(data)

def training_points(filename, batch_size):
    data = parse_log(os.path.join(path, filename))
    pts=[]
    for d in data:
        if 'crossentropy_categorical_1hot_masked' in d  and 'iterations_done' in d:
            pts.append( ( 
                int(d['iterations_done'])*batch_size, 
                float(d['crossentropy_categorical_1hot_masked'])
            ) )
    return pts

#pts = training_points('train.10epochs-8.cpu.log', 8)
#pts = training_points('train.1000epochs-256.gpu.log', 256)
pts = training_points('train.1000epochs-256-adadelta.gpu.log', 256)
#pts

import numpy as np

#from bokeh.embed import notebook_div
#div = notebook_div(plot)

# Or, using the same page's cryptic hints :
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

output_notebook(hide_banner=False)  # If you don't like banners, set to False

#N=1000
#x=np.random.random(size=N)*100
#y=np.random.random(size=N)*100
#radii=np.random.random(size=N)*5.0
#c=[ "#%02x%02x%02x" % (r,g,150) for r,g in zip(np.floor(50+2*x), np.floor(30+2*y)) ]

plot = figure(title="Training Error", plot_height=500, 
              x_axis_label="Sentence Count", y_axis_label="CrossEntropyError")
#plot.circle(x,y, radius=1, fill_color=c, fill_alpha=0.6, line_color=None)
plot.line([p[0]/1000./1000. for p in pts], [np.log(p[1]) for p in pts] )

show(plot)



