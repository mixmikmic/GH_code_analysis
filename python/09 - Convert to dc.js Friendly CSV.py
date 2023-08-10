import sys

sys.path.append('/mnt/home/ubuntu/projects/tools/')

import sys,re,json,os,csv,glob
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import datetime,time,random,traceback
from geopy import distance
import geolocator
geo=geolocator.Geolocator()
geo.init()

files=glob.glob('../data/2014-*/*json')
files.sort()
print len(files)
print files[0]

tweets=[]
for file in files:
# Cycle through files
    fileString=open(file,'r').read().decode('utf-8')
    # Read file as one long string and convert to uniicode
    fileDocs=[json.loads(line) for line in fileString.split('\n')]
    fileDocs=[d for d in fileDocs if d['interaction']['tag_tree']['topic'].keys()[0] in ['Discrimination', 'Prevention']]
    fileTweets=[t for t in fileDocs if t['interaction']['type'] in ['twitter']]
    # Split into lines and load as JSON
    tweets.extend(fileTweets)
    # Add list of tweets from file to global list
print len(tweets)

nTime=0
nId=0
nCity=0
# For counting errors
cities=['Belo Horizonte', u'Brasília, Brasilia', u'Cuiabá', 'Curitiba', 'Fortaleza', 'Manaus', 'Natal, Rio Grande do Norte', 
        'Porto Alegre', 'Recife', 'Rio de Janeiro', 'Salvador, Bahia', u'São Paulo', 'Rio Branco, Acre', u'Maceió', u'Macapá',
        u'Vitória, Espírito Santo', u'Goiânia', u'São Luís, Maranhão', 'Campo Grande, Mato Grosso do Sul', u'Belém, Pará',
        u'João Pessoa, Paraíba', u'Teresina, Piauí', u'Porto Velho, Rondônia', 'Boa Vista, Roraima', u'Florianópolis',
        'Aracaju, Sergipe', 'Palmas, Tocantins']
# Define cities to 'snap' coords to
coords=[]
coords=[geo.geoLocate(c)[0][1:3] for c in cities]
# Get coords from geolocator
tolerance=120
# Set tolerance to snap locations to nearest cities, in KM

outFile=csv.writer(open('cities.csv','w'),delimiter='\t')
for i,j in zip(cities,coords):
    print i,j

print tweets[11]

def getClosestCity(tCoords):
  '''Takes tuple of coordinates, cycles through cities 
     in global variable <cities>, reads their coords from
     global variable <coords> and returns closest
     ------
     returns tuple of coords of closest city,city name
     OR None, if no city within tolerance'''
  dist=999999
  closest='ZZZZ'
  cCoords=[]
  for c,cc in enumerate(cities):
    cDist=distance.distance(tCoords,coords[c])
    if cDist<dist:
      dist=cDist
      closest=cc
      cCoords=coords[c]
  if dist<tolerance:
    return cCoords,closest
  else:
    return None

import gender
g=gender.Gender()
g.gender(tweets[1]['interaction']['author']['name'])

def mungeDate(dummyTime):
  '''Takes Twitter timestamp
     ------
     returns iso format timestamp -> YYY-MM-DD hh:mm:ss
  '''
  # Get from this format: Thu, 02 Jan 2014 16:26:15 +0000...
  timeStruct=datetime.datetime.strptime(dummyTime,'%a, %d %b %Y %H:%M:%S +0000')
  # Gets list with date/time components
  return timeStruct
  # ...into this format mm/DD/YYYYYYY-MM-DD hh:mm:ss

print coords
print coords[cities.index(u'São Paulo')]
getClosestCity(coords[cities.index(u'São Paulo')])

outFile=csv.writer(open('../data/all.csv','w'))
# Open output file
nTime=nId=nCity=nRange=nCategory=nSubCategory=nTopic=0
# Reset error counters

outFile.writerow(['city','lat','lon','origdate','topic']) 

for t,tweet in enumerate(tweets):
  cityCoords=None
  try:
    tTime=tweet['interaction']['created_at']
  except:
    nTime+=1
  try:
    id=tweet['interaction']['id']
  except:
    nId+=1
  try:
    category=tweet['interaction']['tag_tree']['topic'].keys()[0]
  except:
    nCategory+=1
  try:
    subCategory=tweet['interaction']['tag_tree']['topic'].values()[0][0]
  except:
    nSubCategory+=1
  try:
    topic = category + "_" + subCategory
  except:
    nTopic+=1
  if 'geo' in tweet['twitter'].keys():
    res=getClosestCity([tweet['twitter']['geo']['latitude'],tweet['twitter']['geo']['longitude']])   
    if res:
    # If location doesn't snap to chosen cities, within tolerance, then throw away
      (cityCoords,city)=res
      outFile.writerow([city.partition(',')[0].encode("utf-8"),cityCoords[0],cityCoords[1],mungeDate(tTime),topic])
    else:
      nRange+=1
  else:
    nCity+=1
    # print tweet
    # print 'FAILING...'
    # print tweet.keys()
    # sys.exit(1)
    # All these tweets should have lat/long, if not stop and find out why
print nTime,nId,nCity,nRange,nCategory,nSubCategory,nTopic

get_ipython().system('head ../data/all.csv')
get_ipython().system('wc ../data/all.csv')

from IPython.core.display import HTML
styles = open("../css/custom.css", "r").read()
HTML(styles)



