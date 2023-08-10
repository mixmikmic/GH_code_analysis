import pandas as pd, json, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

url='http://en.wikipedia.org/wiki/List_of_airports_in_the_United_Arab_Emirates'
df=pd.read_html(url)
df=df[0].loc[:29].T.set_index(0).T.loc[1:].set_index('IATA[2]')
df

from pygeocoder import Geocoder
apik='AIzaSyDybC2OroTE_XDJTuxjKruxFpby5VDhEGk'

locations={}
for i in df.index:
    if str(i).lower()!='nan':
        try:
            results = Geocoder(apik).geocode(i+' airport UAE')
            locations[i]=results[0].coordinates
            print i
        except:pass

locations.pop('AZI');
locations.pop('DHF');
locations.pop('ZDY');
locations.pop('FJR');

file("locations_ae.json",'w').write(json.dumps(locations))

locations=json.loads(file('locations_ae.json','r').read())

import requests

airportialinks={}
for i in locations:
    print i,
    url='https://cse.google.com/cse?cx=partner-pub-6479063288582225%3A8064105798&cof=FORID%3A10&ie=UTF-8&q='+str(i)+'+airport+united+arab+emirates'
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
                full=url+'arrivals/201703'+str(d)
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

file("mdf_ae_arrv.json",'w').write(json.dumps(mdf.reset_index().to_json()))

len(mdf)

airlines=set(mdf['Airline'])

cities=set(mdf['City'])

file("cities_ae_arrv.json",'w').write(json.dumps(list(cities)))
file("airlines_ae_arrv.json",'w').write(json.dumps(list(airlines)))

citycoords={}

for i in cities:
    if i not in citycoords:
        if i==u'Birmingham': z='Birmingham, UK'
        elif i==u'Valencia': z='Valencia, Spain'
        elif i==u'Naples': z='Naples, Italy'
        elif i==u'St. Petersburg': z='St. Petersburg, Russia'
        elif i==u'Bristol': z='Bristol, UK'
        elif i==u'Victoria': z='Victoria, Seychelles'
        elif i==u'Washington': z='Washington, DC'
        elif i==u'Odessa': z='Odessa, Ukraine'
        else: z=i
        citycoords[i]=Geocoder(apik).geocode(z)
        print i

citysave={}
for i in citycoords:
    citysave[i]={"coords":citycoords[i][0].coordinates,
                 "country":citycoords[i][0].country}

file("citysave_ae_arrv.json",'w').write(json.dumps(citysave))

