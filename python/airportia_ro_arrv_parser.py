import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

url='http://en.wikipedia.org/wiki/List_of_airports_in_Romania'
df=pd.read_html(url)
df=df[0].loc[:17].T.set_index(0).T.loc[2:].set_index('IATA')

df

from pygeocoder import Geocoder
apik='AIzaSyDybC2OroTE_XDJTuxjKruxFpby5VDhEGk'

locations={}
for i in df.index:
    results = Geocoder(apik).geocode(i+' airport romania')
    locations[i]=results[0].coordinates
    print i

file("locations_ro.json",'w').write(json.dumps(locations))

locations=json.loads(file('locations_ro.json','r').read())

import requests

airportialinks={}
for i in locations:
    print i,
    url='https://cse.google.com/cse?cx=partner-pub-6479063288582225%3A8064105798&cof=FORID%3A10&ie=UTF-8&q='+str(i)+'+airport+romania'
    m=requests.get(url).content
    z=pd.read_html(m)[5][0][0]
    z=z[z.find('http'):]
    airportialinks[i]=z
    print z

#reformat
for z in airportialinks:
    airportialinks[z]=airportialinks[z].split('arrivals')[0].split('departures')[0].replace(' ','').replace('...','-international-')
    if airportialinks[z][-1]!='/':airportialinks[z]+='/' 
    #manual fixes
    if z=='TSR':airportialinks[z]='https://www.airportia.com/romania/timişoara-traian-vuia-airport/'
    print airportialinks[z]

sch={}

for i in locations:
    print i
    if i not in sch:sch[i]={}
    if i!='TGM':
        #march 11-24 = 2 weeks
        for d in range (11,25):
            if d not in sch[i]:
                try:
                    url=airportialinks[i]
                    full=url+'arrivals/201703'+str(d)
                    m=requests.get(full).content
                    sch[i][full]=pd.read_html(m)[0]
                    #print full
                except: pass #print 'no tables',i,d
    else:
        #november 17-30 = 2 weeks
        for d in range (17,31):
            if d not in sch[i]:
                try:
                    url=airportialinks[i]
                    full=url+'arrivals/201611'+str(d)
                    m=requests.get(full).content
                    sch[i][full]=pd.read_html(m)[0]
                    #print full
                except: pass #print 'no tables',i,d

mdf=pd.DataFrame()

for i in sch:
    for d in sch[i]:
        df=sch[i][d].drop(sch[i][d].columns[3:],axis=1).drop(sch[i][d].columns[0],axis=1)
        df['To']=i
        df['Date']=d
        mdf=pd.concat([mdf,df])

mdf=mdf.replace('Hahn','Frankfurt')
mdf=mdf.replace('Hahn HHN','Frankfurt HHN')

mdf['City']=[i[:i.rfind(' ')] for i in mdf['From']]
mdf['Airport']=[i[i.rfind(' ')+1:] for i in mdf['From']]

file("mdf_ro_arrv.json",'w').write(json.dumps(mdf.reset_index().to_json()))

len(mdf)

airlines=set(mdf['Airline'])

cities=set(mdf['City'])

file("cities_ro_arrv.json",'w').write(json.dumps(list(cities)))
file("airlines_ro_arrv.json",'w').write(json.dumps(list(airlines)))

citycoords={}

for i in cities:
    if i not in citycoords:
        if i==u'Birmingham': z='Birmingham, UK'
        elif i==u'Valencia': z='Valencia, Spain'
        elif i==u'Naples': z='Naples, Italy'
        elif i==u'St. Petersburg': z='St. Petersburg, Russia'
        elif i==u'Bristol': z='Bristol, UK'
        else: z=i
        citycoords[i]=Geocoder(apik).geocode(z)
        print i

citysave={}
for i in citycoords:
    citysave[i]={"coords":citycoords[i][0].coordinates,
                 "country":citycoords[i][0].country}

file("citysave_ro_arrv.json",'w').write(json.dumps(citysave))

