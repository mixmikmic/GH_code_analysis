import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from pygeocoder import Geocoder
apik='AIzaSyDybC2OroTE_XDJTuxjKruxFpby5VDhEGk'

SD={}
citycoords={}
clustercoords={}
error=[]
cerror=[]
SC=json.loads(file('../json/SC2.json','r').read())
cluster=json.loads(file('../json/cluster.json','r').read())

for c in SC:
    print c,
    #country not parsed yet
    for i in SC[c]:
        if cluster[i] not in clustercoords:
            print cluster[i],
            z=cluster[i]+', '+c
            try: clustercoords[cluster[i]]=Geocoder(apik).geocode(z)
            except: cerror.append(cluster[i])
        if i not in citycoords:
            print i,
            x=i+' airport, '+cluster[i]+', '+c
            try: citycoords[i]=Geocoder(apik).geocode(x)
            except: error.append(i)
    print

print len(citycoords),len(error),len(clustercoords),len(cerror)

citysave=json.loads(file('../json/citysave.json','r').read())

for i in citycoords:
    citysave[i]['coords']=citycoords[i][0].coordinates
    citysave[i]['country']=citycoords[i][0].country

file("../json/citysave2.json",'w').write(json.dumps(citysave))

#load if saved
citysave=json.loads(file('../json/citysave2.json','r').read())

cities=set()
for c in SC:
    for i in SC[c]:
        cities.add(i)

for i in set(citysave.keys()):
    if i not in cities:
        print i,
        citysave.pop(i)

for i in citysave:
    citysave[i]['city']=cluster[i]

file("../json/citysave3.json",'w').write(json.dumps(citysave))

I3=json.loads(file('../json/I3.json','r').read())
cnc_path='../../universal/countries/'
cnc=pd.read_csv(cnc_path+'cnc.csv',header=1).set_index('ISO2')

cnc

for i in citysave:
    country=citysave[i]['country']
    if country==None:print i,citysave[i]['city']
    #citysave[i]['iso2']=cnc.loc['ISO2'][country]
    #citysave[i]['pretty']=cnc.loc['pretty'][country]

#manual fix
citysave['IXJ']['country']='India'
citysave['IXL']['country']='India'
citysave['PRN']['country']='Kosovo'
citysave['AWK']['country']='United States'
citysave['SXR']['country']='India'
citysave['KDU']['country']='Pakistan'
citysave['GIL']['country']='Pakistan'

for c in SC:
    for i in SC[c]:
        if i in citysave:
            if citysave[i]['country']!=c:
                citysave[i]['country']=c
                print i,citysave[i]['city'],':',citysave[i]['country'],c

for i in citysave:
    country=I3[citysave[i]['country']]
    citysave[i]['iso2']=country
    citysave[i]['pretty']=cnc.loc['pretty'][country]

file("../json/citysave4.json",'w').write(json.dumps(citysave))

unicities={}
for i in cluster:
    if cluster[i] not in unicities:
        unicities[cluster[i]]=citysave[i]['country']
citycoords={}
for i in cluster:
    if cluster[i] not in citycoords:
        citycoords[citysave[i]['city']]=citysave[i]['coords']

file('../json/citycoords.json','w').write(json.dumps(citycoords))
file('../json/unicities.json','w').write(json.dumps(unicities))

print len(citysave),len(citycoords)

