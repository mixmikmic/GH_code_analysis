import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

url='http://en.wikipedia.org/wiki/List_of_airports_in_Jordan'
df=pd.read_html(url)
df=df[0].loc[:3].T.set_index(0).T.loc[1:].set_index('IATA')

df

from pygeocoder import Geocoder
apik='AIzaSyDybC2OroTE_XDJTuxjKruxFpby5VDhEGk'

locations={}
for i in df.index:
    results = Geocoder(apik).geocode(i+' airport Jordan')
    locations[i]=results[0].coordinates
    print i

locations.pop('ADJ')

file("locations_jo.json",'w').write(json.dumps(locations))

locations=json.loads(file('locations_jo.json','r').read())

import requests

airportialinks={}
for i in locations:
    print i,
    url='https://cse.google.com/cse?cx=partner-pub-6479063288582225%3A8064105798&cof=FORID%3A10&ie=UTF-8&q='+str(i)+'+airport+jordan'
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
    if z=='AQJ':airportialinks[z]=u'https://www.airportia.com/jordan/aqaba-king-hussein-international-airport/'
    print airportialinks[z]

sch={}

for i in locations:
    print i
    if i not in sch:sch[i]={}
    #march 11-24 = 2 weeks
    for d in range (11,25):
        if d not in sch[i]:
            try:
                url=airportialinks[i]
                full=url+'departures/201703'+str(d)
                m=requests.get(full).content
                sch[i][full]=pd.read_html(m)[0]
                #print full
            except: pass #print 'no tables',i,d

for i in range(11,25):
    testurl=u'https://www.airportia.com/jordan/queen-alia-international-airport/departures/201703'+str(i)
    print 'nr. of flights on March',i,':',len(sch['AMM'][testurl])
testurl=u'https://www.airportia.com/jordan/queen-alia-international-airport/departures/20170318'
k=sch['AMM'][testurl]
k[k['To']=='Frankfurt FRA']

mdf=pd.DataFrame()

for i in sch:
    for d in sch[i]:
        df=sch[i][d].drop(sch[i][d].columns[3:],axis=1).drop(sch[i][d].columns[0],axis=1)
        df['From']=i
        df['Date']=d
        mdf=pd.concat([mdf,df])

mdf['City']=[i[:i.rfind(' ')] for i in mdf['To']]
mdf['Airport']=[i[i.rfind(' ')+1:] for i in mdf['To']]

k=mdf[mdf['Date']==testurl]
k[k['To']=='Frankfurt FRA']

file("mdf_jo_dest.json",'w').write(json.dumps(mdf.reset_index().to_json()))

len(mdf)

airlines=set(mdf['Airline'])

cities=set(mdf['City'])

file("cities_jo_dest.json",'w').write(json.dumps(list(cities)))
file("airlines_jo_dest.json",'w').write(json.dumps(list(airlines)))

citycoords={}

for i in cities:
    if i not in citycoords:
        if i==u'Birmingham': z='Birmingham, UK'
        elif i==u'Valencia': z='Valencia, Spain'
        elif i==u'Naples': z='Naples, Italy'
        elif i==u'St. Petersburg': z='St. Petersburg, Russia'
        elif i==u'Bristol': z='Bristol, UK'
        elif i==u'Beida': z='Bayda, Libya'
        else: z=i
        citycoords[i]=Geocoder(apik).geocode(z)
        print i

citysave={}
for i in citycoords:
    citysave[i]={"coords":citycoords[i][0].coordinates,
                 "country":citycoords[i][0].country}

file("citysave_jo_dest.json",'w').write(json.dumps(citysave))

