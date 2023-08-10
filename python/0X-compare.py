import holoviews as hv
hv.extension('bokeh','matplotlib')
from IPython.core import debugger
ist = debugger.set_trace

import os
import pandas as pd
import pstats


energies = []
names = []
files = [f for f in os.listdir('./energy/') if f[0] != '.']
for file in files:
    names.append(os.path.splitext(os.path.basename(file))[0])
    with open('./energy/' + file,'r') as f:
        energies.append(float(f.readline()))
es = pd.Series(energies,index=names,name='Energy')   


times = []
time_names = []
mem_names = []
mems = []
files = [f for f in os.listdir('./prof/') if f[0] != '.']
for file in files:
    fname = './prof/' + file
    name,ext = os.path.splitext(os.path.basename(file))
    if ext == '.prof':
        time_names.append(name)
        times.append(pstats.Stats(fname).total_tt)
    elif ext == '.memprof':
        mem_names.append(name)
        with open(fname,'r') as f:
            mems.append([float(i) for i in f.readlines()])
ts = pd.Series(times,index=time_names,name='RunTime') 
ms1 = pd.Series([i[0] for i in mems],index=mem_names,name='MaxMem')   
ms2 = pd.Series([i[1] for i in mems],index=mem_names,name='IncMem')   

df = pd.concat([es,ts,ms1,ms2],axis=1).sort_values(by='RunTime')

#Add in relative runtime column
df.insert(2,column='RelTime',value=df['RunTime']/df['RunTime'][df.index=='python'][0])
df

get_ipython().run_cell_magic('opts', "Bars [xrotation=40,height=400,width=600,show_grid=True,tools=['hover']]", "%%opts Bars [fontsize={'ticks':14,'labels':16}] (alpha=0.6)\n\nhvb1 = (hv.Bars(df.reset_index(),kdims=['index'],vdims=['RelTime'])\n         .redim.label(RelTime = 'Relative Time',index='Approach'))\n\nhvb2 = (hv.Bars(df.reset_index(),kdims=['index'],vdims=['RunTime'])\n        .redim.label(RunTime = 'Run Time',index='Approach')\n        .redim.unit(RunTime = 's'))\n\nhvb3 = (hv.Bars(df.reset_index(),kdims=['index'],vdims=['MaxMem'])\n        .redim.label(MaxMem = 'Maximum Memory',index='Approach')\n        .redim.unit(MaxMem = 'MB'))\n\n(hvb1 + hvb2 + hvb3).cols(1)")





