import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cluster=json.loads(file('../json/cluster.json','r').read())
citysave=json.loads(file('../json/citysave2.json','r').read())
pop_countries=json.loads(file('../json/pop_countries2.json','r').read())
pop_cities=json.loads(file('../json/pop_cities.json','r').read())

unicities={}
for i in cluster:
    if cluster[i] not in unicities:
        unicities[cluster[i]]=citysave[i]['country']

parent={}

for i in pop_cities:
    #if a k times larger city is within x km
    k=4
    x=100
    ct={}
    for j in pop_cities[i]['nearby']:
        if pop_cities[i]['nearby'][j]['people']>pop_cities[i]['pop']*k:
            if pop_cities[i]['nearby'][j]['km']<x:
                ct[pop_cities[i]['nearby'][j]['people']]=j
    if ct:        
        cm=ct[max(ct.keys())]
        parent[i]={cm:pop_cities[i]['nearby'][cm]['people']}
    else:
        parent[i]={i:pop_cities[i]['pop']}

