import requests
from shapely.geometry import shape, Point

r = requests.get('https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&format=GeoJSON')

for feature in r.json()['features']:
    if feature['properties']['community'] == 'AUSTIN':
        austin = feature

poly = shape(austin['geometry'])

import os

def get_data(table):
    r = requests.get('%stable/json/%s' % (os.environ['NEWSROOMDB_URL'], table))
    return r.json()

shootings = get_data('shootings')
homicides = get_data('homicides')

shootings_ca = []

for row in shootings:
    if not row['Geocode Override']:
        continue
    points = row['Geocode Override'][1:-1].split(',')
    if len(points) != 2:
        continue
    point = Point(float(points[1]), float(points[0]))
    row['point'] = point
    if poly.contains(point):
        shootings_ca.append(row)

print 'Found %d shootings in this community area' % len(shootings_ca)
for f in shootings_ca:
    print f['Date'], f['Time'],  f['Age'], f['Sex'], f['Shooting Location']

homicides_ca = []
years = {}

for row in homicides:
    if not row['Geocode Override']:
        continue
    points = row['Geocode Override'][1:-1].split(',')
    if len(points) != 2:
        continue
    point = Point(float(points[1]), float(points[0]))
    row['point'] = point
    if poly.contains(point):
        homicides_ca.append(row)

print 'Found %d homicides in this community area' % len(homicides_ca)
for f in homicides_ca:
    print f['Occ Date'], f['Occ Time'],  f['Age'], f['Sex'], f['Address of Occurrence']
    if not f['Occ Date']:
        continue
    dt = datetime.strptime(f['Occ Date'], '%Y-%m-%d')
    if dt.year not in years:
        years[dt.year] = 0
    years[dt.year] += 1
print years

import pyproj
from datetime import datetime, timedelta

geod = pyproj.Geod(ellps='WGS84')
associated = []

for homicide in homicides_ca:
    if not homicide['Occ Time']:
        homicide['Occ Time'] = '00:01'
    if not homicide['Occ Date']:
        homicide['Occ Date'] = '2000-01-01'
    homicide_dt = datetime.strptime('%s %s' % (homicide['Occ Date'], homicide['Occ Time']), '%Y-%m-%d %H:%M')
    for shooting in shootings_ca:
        if not shooting['Time']:
            shooting['Time'] = '00:01'
        if not shooting['Time']:
            shooting['Time'] = '2000-01-01'
        shooting_dt = datetime.strptime('%s %s' % (shooting['Date'], shooting['Time']), '%Y-%m-%d %H:%M')
        diff = homicide_dt - shooting_dt
        seconds = divmod(diff.days * 86400 + diff.seconds, 60)[0]
        if abs(seconds) <= 600:
            angle1, angle2, distance = geod.inv(
                homicide['point'].x, homicide['point'].y, shooting['point'].x, shooting['point'].y)
            if distance < 5:
                associated.append((homicide, shooting))
                break
print len(associated)

years = {}

for homicide in homicides:
    if not homicide['Occ Date']:
        continue
    dt = datetime.strptime(homicide['Occ Date'], '%Y-%m-%d')
    if dt.year not in years:
        years[dt.year] = 0
    years[dt.year] += 1

print years

from csv import DictWriter
from ftfy import fix_text, guess_bytes

for idx, row in enumerate(shootings_ca):
    if 'point' in row.keys():
        del row['point']
    for key in row:
        #print idx, key, row[key]
        if type(row[key]) is str:
            #print row[key]
            row[key] = fix_text(row[key].replace('\xa0', '').decode('utf8'))

for idx, row in enumerate(homicides_ca):
    if 'point' in row.keys():
        del row['point']
    for key in row:
        #print idx, key, row[key]
        if type(row[key]) is str:
            #print row[key]
            row[key] = row[key].decode('utf8')


with open('/Users/abrahamepton/Documents/austin_shootings.csv', 'w+') as fh:
    writer = DictWriter(fh, sorted(shootings_ca[0].keys()))
    writer.writeheader()
    for row in shootings_ca:
        try:
            writer.writerow(row)
        except:
            print row

with open('/Users/abrahamepton/Documents/austin_homicides.csv', 'w+') as fh:
    writer = DictWriter(fh, sorted(homicides_ca[0].keys()))
    writer.writeheader()
    for row in homicides_ca:
        try:
            writer.writerow(row)
        except:
            print row



