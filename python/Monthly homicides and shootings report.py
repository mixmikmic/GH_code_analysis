import os
import requests

def get_table_data(table_name):
    url = '%stable/json/%s' % (os.environ['NEWSROOMDB_URL'], table_name)
    try:
        r = requests.get(url)
        return r.json()
    except:
        print 'doh'
        return get_table_data(table_name)

homicides = get_table_data('homicides')
shootings = get_table_data('shootings')

print 'Found %d homicides and %d shootings' % (len(homicides), len(shootings))

from datetime import date, datetime
today = date.today()

homicides_this_month = {}
for h in homicides:
    try:
        dt = datetime.strptime(h['Occ Date'], '%Y-%m-%d')
    except ValueError:
        continue
    if dt.month == today.month:
        if dt.year not in homicides_this_month:
            homicides_this_month[dt.year] = []
        homicides_this_month[dt.year].append(h)

shootings_this_month = {}
for s in shootings:
    try:
        dt = datetime.strptime(s['Date'], '%Y-%m-%d')
    except ValueError:
        continue
    if dt.month == today.month:
        if dt.year not in shootings_this_month:
            shootings_this_month[dt.year] = []
        shootings_this_month[dt.year].append(s)

for year in sorted(shootings_this_month.keys(), reverse=True):
    try:
        s = len(shootings_this_month[year])
    except:
        s = 0
    try:
        h = len(homicides_this_month[year])
    except:
        h = 0
    print '%d:\t%d shootings\t\t%d homicides' % (year, s, h)

from datetime import date, timedelta

test_date = date.today()
one_day = timedelta(days=1)

shooting_days = {}
for shooting in shootings:
    if shooting['Date'] not in shooting_days:
        shooting_days[shooting['Date']] = 0
    shooting_days[shooting['Date']] += 1

while test_date.year > 2013:
    if test_date.strftime('%Y-%m-%d') not in shooting_days:
        print 'No shootings on %s' % test_date
    test_date -= one_day

from datetime import date, timedelta

test_date = date.today()
one_day = timedelta(days=1)

homicide_days = {}
for homicide in homicides:
    if homicide['Occ Date'] not in homicide_days:
        homicide_days[homicide['Occ Date']] = 0
    homicide_days[homicide['Occ Date']] += 1

while test_date.year > 2013:
    if test_date.strftime('%Y-%m-%d') not in homicide_days:
        print 'No homicides on %s' % test_date
    test_date -= one_day

coordinates = []
for homicide in homicides:
    if not homicide['Occ Date'].startswith('2015-'):
        continue
    # Since the format of this field is (x, y) (or y, x? I always confuse the two) we need to extract just x and y
    try:
        coordinates.append(
            (homicide['Geocode Override'][1:-1].split(',')[0], homicide['Geocode Override'][1:-1].split(',')[1]))
    except:
        # Not valid/expected lat/long format
        continue

print len(coordinates)

for coordinate in coordinates:
    print '%s,%s' % (coordinate[0].strip(), coordinate[1].strip())



