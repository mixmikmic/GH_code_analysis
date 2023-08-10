import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from pygeocoder import Geocoder
apik='AIzaSyDybC2OroTE_XDJTuxjKruxFpby5VDhEGk'

locations=json.loads(file('locations_jo.json','r').read())

mdf_dest=pd.read_json(json.loads(file('mdf_jo_dest.json','r').read()))
mdf_arrv=pd.read_json(json.loads(file('mdf_jo_arrv.json','r').read()))

citysave_dest=json.loads(file('citysave_jo_dest.json','r').read())
citysave_arrv=json.loads(file('citysave_jo_arrv.json','r').read())

mdf_dest['ID']=mdf_dest['From']
mdf_dest.head()

mdf_arrv['ID']=mdf_arrv['To']
mdf_arrv.head()

mdf=pd.concat([mdf_dest,mdf_arrv])

len(mdf_dest)

len(mdf_arrv)

mdf

mdg=mdf.set_index(['ID','City','Airport','Airline'])

k=mdg.loc['AMM'].loc['Frankfurt'].loc['FRA']
testurl=u'https://www.airportia.com/jordan/queen-alia-international-airport/departures/20170318'
k[k['Date']==testurl]

k=mdg.loc['AMM'].loc['Frankfurt'].loc['FRA']
testurl=u'https://www.airportia.com/jordan/queen-alia-international-airport/arrivals/20170318'
k[k['Date']==testurl]

k=mdg.loc['AMM'].loc['Frankfurt'].loc['FRA']
for i in range(11,25):
    testurl=u'https://www.airportia.com/jordan/queen-alia-international-airport/departures/201703'+str(i)
    print 'AMM-FRA March',i, 'departures',len(k[k['Date']==testurl]),
    testurl=u'https://www.airportia.com/jordan/queen-alia-international-airport/arrivals/201703'+str(i)
    print 'arrivals', len(k[k['Date']==testurl])

len(k)/14

flights={}
minn=1.0 #want to see minimum 1 flight in the past 2 weeks
for i in mdg.index.get_level_values(0).unique():
    #2 weeks downloaded. want to get weekly freq. but multi by 2 dept+arrv
    d=4.0
    if i not in flights:flights[i]={}
    for j in mdg.loc[i].index.get_level_values(0).unique():
        if len(mdg.loc[i].loc[j])>minn: #minimum 1 flights required in this period at least once every 2 weeks
            if j not in flights[i]:flights[i][j]={'airports':{},'7freq':0}
            flights[i][j]['7freq']=len(mdg.loc[i].loc[j])/d 
            for k in mdg.loc[i].loc[j].index.get_level_values(0).unique():
                if len(mdg.loc[i].loc[j].loc[k])>minn:
                    if k not in flights[i][j]['airports']:flights[i][j]['airports'][k]={'airlines':{},'7freq':0}
                    flights[i][j]['airports'][k]['7freq']=len(mdg.loc[i].loc[j].loc[k])/d
                    for l in mdg.loc[i].loc[j].loc[k].index.get_level_values(0).unique():
                        try:
                            if len(mdg.loc[i].loc[j].loc[k].loc[l])>minn: 
                                if l not in flights[i][j]['airports'][k]['airlines']:flights[i][j]['airports'][k]['airlines'][l]={'7freq':0}
                                flights[i][j]['airports'][k]['airlines'][l]['7freq']=len(mdg.loc[i].loc[j].loc[k].loc[l])/d
                        except:pass

file("flights_jo.json",'w').write(json.dumps(flights))

