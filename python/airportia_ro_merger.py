import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from pygeocoder import Geocoder
apik='AIzaSyDybC2OroTE_XDJTuxjKruxFpby5VDhEGk'

locations=json.loads(file('locations_ro.json','r').read())

mdf_dest=pd.read_json(json.loads(file('mdf_ro_dest.json','r').read()))
mdf_arrv=pd.read_json(json.loads(file('mdf_ro_arrv.json','r').read()))

citysave_dest=json.loads(file('citysave_ro_dest.json','r').read())
citysave_arrv=json.loads(file('citysave_ro_arrv.json','r').read())

mdf_dest['ID']=mdf_dest['From']
mdf_dest.head()

mdf_arrv['ID']=mdf_arrv['To']
mdf_arrv.head()

mdf=pd.concat([mdf_dest,mdf_arrv])

mdf

mdg=mdf.set_index(['ID','City','Airport','Airline'])

len(mdg)

flights={}
minn=1.0
for i in mdg.index.get_level_values(0).unique():
    #2 weeks downloaded. want to get weekly freq. but multi by 2 dept+arrv
    d=4.0
    if i not in flights:flights[i]={}
    for j in mdg.loc[i].index.get_level_values(0).unique():
        if len(mdg.loc[i].loc[j])>minn: #minimum 1 flights required in this period once every 2 weeks
            if j not in flights[i]:flights[i][j]={'airports':{},'7freq':0}
            flights[i][j]['7freq']=len(mdg.loc[i].loc[j])/d 
            for k in mdg.loc[i].loc[j].index.get_level_values(0).unique():
                if len(mdg.loc[i].loc[j].loc[k])>minn:
                    if k not in flights[i][j]['airports']:flights[i][j]['airports'][k]={'airlines':{},'7freq':0}
                    flights[i][j]['airports'][k]['7freq']=len(mdg.loc[i].loc[j].loc[k])/d
                    for l in mdg.loc[i].loc[j].loc[k].index.get_level_values(0).unique():
                        if len(mdg.loc[i].loc[j].loc[k].loc[l])>minn: 
                            if l not in flights[i][j]['airports'][k]['airlines']:flights[i][j]['airports'][k]['airlines'][l]={'7freq':0}
                            flights[i][j]['airports'][k]['airlines'][l]['7freq']=len(mdg.loc[i].loc[j].loc[k].loc[l])/d

flights['TGM']['Budapest']=flights['CLJ']['Budapest']

for j in flights['TGM']:
    if flights['CLJ'][j]['7freq']-flights['TGM'][j]['7freq']>0:
        flights['CLJ'][j]['7freq']-=flights['TGM'][j]['7freq']
        ap=list(flights['TGM'][j]['airports'].keys())[0]
        flights['CLJ'][j]['airports'][ap]['7freq']-=flights['TGM'][j]['7freq']
        flights['CLJ'][j]['airports'][ap]['airlines'][u'Wizz Air']['7freq']-=flights['TGM'][j]['7freq']
    else: flights['CLJ'].pop(j)

file("flights_ro.json",'w').write(json.dumps(flights))

