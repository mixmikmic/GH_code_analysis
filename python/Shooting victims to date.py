import os
import requests

# A big object to hold all our data between steps
data = {}

def get_table_url(table_name, base_url=os.environ['NEWSROOMDB_URL']):
    return '{}table/json/{}'.format(os.environ['NEWSROOMDB_URL'], table_name)

def get_table_data(table_name):
    url = get_table_url(table_name)
    
    try:
        r = requests.get(url)
        return r.json()
    except:
        print("Request failed. Probably because the response is huge.  We should fix this.")
        return get_table_data(table_name)

data['shooting_victims'] = get_table_data('shootings')

print("Loaded {} shooting victims".format(len(data['shooting_victims'])))

from datetime import date, datetime

def get_shooting_date(shooting_victim):
    return datetime.strptime(shooting_victim['Date'], '%Y-%m-%d')

def shooting_is_ytd(shooting_victim, today):
    try:
        shooting_date = get_shooting_date(shooting_victim)
    except ValueError:
        if shooting_victim['RD Number']:
            msg = "Could not parse date for shooting victim with RD Number {}".format(
                shooting_victim['RD Number'])
        else:
            msg = "Could not parse date for shooting victim with record ID {}".format(
                shooting_victim['_id'])
        
        print(msg)
        return False
        
    return (shooting_date.month <= today.month and
            shooting_date.day <= today.day)

today = date(2016, 3, 30)
#today = date.today()

# Use a list comprehension to filter the shooting victims to ones that
# occured on or before today's month and day.
# Also sort by date because it makes it easier to group by year
data['shooting_victims_ytd'] = sorted([sv for sv in data['shooting_victims']
                                       if shooting_is_ytd(sv, today)],
                                      key=get_shooting_date)

import itertools

def get_shooting_year(shooting_victim):
    shooting_date = get_shooting_date(shooting_victim)
    return shooting_date.year

data['shooting_victims_ytd_by_year'] = []

for year, grp in itertools.groupby(data['shooting_victims_ytd'], key=get_shooting_year):
    data['shooting_victims_ytd_by_year'].append((year, list(grp)))

data['shooting_victims_ytd_by_year_totals'] = [(year, len(shooting_victims))
                                               for year, shooting_victims
                                               in data['shooting_victims_ytd_by_year']]

import csv
import sys

writer = csv.writer(sys.stdout)
writer.writerow(['year', 'num_shooting_victims'])

for year, num_shooting_victims in data['shooting_victims_ytd_by_year_totals']:
    writer.writerow([year, num_shooting_victims])

shooting_victims_2016 = next(shooting_victims
                             for year, shooting_victims
                             in data['shooting_victims_ytd_by_year']
                             if year == 2016)
num_shooting_victims_2016 = next(num_shooting_victims
                                 for year, num_shooting_victims
                                 in data['shooting_victims_ytd_by_year_totals']
                                 if year == 2016)
today = date.today()
num_shootings = 0
for shooting_victim in shooting_victims_2016:
    num_shootings += 1
    shooting_date = get_shooting_date(shooting_victim)
    assert shooting_date.year == 2016
    assert shooting_date.month <= today.month
    assert shooting_date.day <= today.day
    
assert num_shootings == num_shooting_victims_2016



