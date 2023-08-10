few_crashes_url = 'http://www.arcgis.com/sharing/rest/content/items/5a8841f92e4a42999c73e9a07aca0c23/data?f=json&token=lddNjwpwjOibZcyrhJiogNmyjIZmzh-pulx7jPD9c559e05tWo6Qr8eTcP7Deqw_CIDPwZasbNOCSBHfthynf-8WRMmguxHbIFptbZQvnpRupJHSY8Abrz__xUteBS93MitgvoU6AqSN5eDVKRYiUg..'
removed_url = 'http://www.arcgis.com/sharing/rest/content/items/1e01ac5dc4d54dc186502316feab156e/data?f=json&token=lddNjwpwjOibZcyrhJiogNmyjIZmzh-pulx7jPD9c559e05tWo6Qr8eTcP7Deqw_CIDPwZasbNOCSBHfthynf-8WRMmguxHbIFptbZQvnpRupJHSY8Abrz__xUteBS93MitgvoU6AqSN5eDVKRYiUg..'

import requests
def extract_features(url, title=None):
    r = requests.get(url)
    idx = 0
    found = False
    if title:
        while idx < len(r.json()['operationalLayers']):
            for item in r.json()['operationalLayers'][idx].items():
                if item[0] == 'title' and item[1] == title:
                    found = True
                    break
            if found:
                break
            idx += 1
    try:
        return r.json()['operationalLayers'][idx]['featureCollection']['layers'][0]['featureSet']['features']
    except IndexError, e:
        return {}

few_crashes = extract_features(few_crashes_url)
all_cameras = extract_features(removed_url, 'All Chicago red light cameras')
removed_cameras = extract_features(removed_url, 'red-light-cams')
print 'Found %d data points for few-crash intersections, %d total cameras and %d removed camera locations' % (
    len(few_crashes), len(all_cameras), len(removed_cameras))

filtered_few_crashes = [
    point for point in few_crashes if point['attributes']['LONG_X'] != 0 and point['attributes']['LAT_Y'] != 0]

cameras = {}
for point in all_cameras:
    label = point['attributes']['LABEL']
    if label not in cameras:
        cameras[label] = point
        cameras[label]['attributes']['Few crashes'] = False
        cameras[label]['attributes']['To be removed'] = False

for point in filtered_few_crashes:
    label = point['attributes']['LABEL']
    if label not in cameras:
        print 'Missing label %s' % label
    else:
        cameras[label]['attributes']['Few crashes'] = True

for point in removed_cameras:
    label = point['attributes']['displaylabel'].replace(' and ', '-')
    if label not in cameras:
        print 'Missing label %s' % label
    else:
        cameras[label]['attributes']['To be removed'] = True

counter = {
    'both': {
        'names': [],
        'count': 0
    },
    'crashes only': {
        'names': [],
        'count': 0
    },
    'removed only': {
        'names': [],
        'count': 0
    }
}

for camera in cameras:
    if cameras[camera]['attributes']['Few crashes']:
        if cameras[camera]['attributes']['To be removed']:
            counter['both']['count'] += 1
            counter['both']['names'].append(camera)
        else:
            counter['crashes only']['count'] += 1
            counter['crashes only']['names'].append(camera)
    elif cameras[camera]['attributes']['To be removed']:
        counter['removed only']['count'] += 1
        counter['removed only']['names'].append(camera)

print '%d locations had few crashes and were slated to be removed: %s\n' % (
    counter['both']['count'], '; '.join(counter['both']['names']))
print '%d locations had few crashes but were not slated to be removed: %s\n' % (
    counter['crashes only']['count'], '; '.join(counter['crashes only']['names']))
print '%d locations were slated to be removed despite having reasonable numbers of crashes: %s' % (
    counter['removed only']['count'], '; '.join(counter['removed only']['names']))

from csv import DictReader
from StringIO import StringIO

data_portal_url = 'https://data.cityofchicago.org/api/views/thvf-6diy/rows.csv?accessType=DOWNLOAD'
r = requests.get(data_portal_url)
fh = StringIO(r.text)
reader = DictReader(fh)

def cleaner(str):
    filters = [
        ('Stony?Island', 'Stony Island'),
        ('Van?Buren', 'Van Buren'),
        (' (SOUTH INTERSECTION)', '')
    ]
    for filter in filters:
        str = str.replace(filter[0], filter[1])
    return str

for line in reader:
    line['INTERSECTION'] = cleaner(line['INTERSECTION'])
    cameras[line['INTERSECTION']]['attributes']['current'] = line

counter = {
    'not current': [],
    'current': [],
    'not current and slated for removal': [],
    'not current and not slated for removal': [],
    'current and slated for removal': []
}
for camera in cameras:
    if 'current' not in cameras[camera]['attributes']:
        counter['not current'].append(camera)
        if cameras[camera]['attributes']['To be removed']:
            counter['not current and slated for removal'].append(camera)
        else:
            counter['not current and not slated for removal'].append(camera)
    else:
        counter['current'].append(camera)
        if cameras[camera]['attributes']['To be removed']:
            counter['current and slated for removal'].append(camera)

for key in counter:
    print key, len(counter[key])
    print '; '.join(counter[key]), '\n'

import requests
from csv import DictReader
from datetime import datetime
from StringIO import StringIO

data_portal_url = 'https://data.cityofchicago.org/api/views/spqx-js37/rows.csv?accessType=DOWNLOAD'
r = requests.get(data_portal_url)
fh = StringIO(r.text)
reader = DictReader(fh)

def violation_cleaner(str):
    filters = [
        (' AND ', '-'),
        (' and ', '-'),
        ('/', '-'),
        # These are streets spelled one way in ticket data, another way in location data
        ('STONEY ISLAND', 'STONY ISLAND'),
        ('CORNELL DRIVE', 'CORNELL'),
        ('NORTHWEST HWY', 'NORTHWEST HIGHWAY'),
        ('CICERO-I55', 'CICERO-STEVENSON NB'),
        ('31ST ST-MARTIN LUTHER KING DRIVE', 'DR MARTIN LUTHER KING-31ST'),
        ('4700 WESTERN', 'WESTERN-47TH'),
        ('LAKE SHORE DR-BELMONT', 'LAKE SHORE-BELMONT'),
        # These are 3-street intersections where the ticket data has 2 streets, location data has 2 other streets
        ('KIMBALL-DIVERSEY', 'MILWAUKEE-DIVERSEY'),
        ('PULASKI-ARCHER', 'PULASKI-ARCHER-50TH'),
        ('KOSTNER-NORTH', 'KOSTNER-GRAND-NORTH'),
        ('79TH-KEDZIE', 'KEDZIE-79TH-COLUMBUS'),
        ('LINCOLN-MCCORMICK', 'KIMBALL-LINCOLN-MCCORMICK'),
        ('KIMBALL-LINCOLN', 'KIMBALL-LINCOLN-MCCORMICK'),
        ('DIVERSEY-WESTERN', 'WESTERN-DIVERSEY-ELSTON'),
        ('HALSTED-FULLERTON', 'HALSTED-FULLERTON-LINCOLN'),
        ('COTTAGE GROVE-71ST', 'COTTAGE GROVE-71ST-SOUTH CHICAGO'),
        ('DAMEN-FULLERTON', 'DAMEN-FULLERTON-ELSTON'),
        ('DAMEN-DIVERSEY', 'DAMEN-DIVERSEY-CLYBOURN'),
        ('ELSTON-FOSTER', 'ELSTON-LAPORTE-FOSTER'),
        ('STONY ISLAND-79TH', 'STONY ISLAND-79TH-SOUTH CHICAGO'),
        # This last one is an artifact of the filter application process
        ('KIMBALL-LINCOLN-MCCORMICK-MCCORMICK', 'KIMBALL-LINCOLN-MCCORMICK')
    ]
    for filter in filters:
        str = str.replace(filter[0], filter[1])
    return str

def intersection_is_reversed(key, intersection):
    split_key = key.upper().split('-')
    split_intersection = intersection.upper().split('-')
    if len(split_key) != len(split_intersection):
        return False
    for k in split_key:
        if k not in split_intersection:
            return False
    for k in split_intersection:
        if k not in split_key:
            return False
    return True
    

missing_intersections = set()
for idx, line in enumerate(reader):
    line['INTERSECTION'] = violation_cleaner(line['INTERSECTION'])
    found = False
    for key in cameras:
        if key.lower() == line['INTERSECTION'].lower() or intersection_is_reversed(key, line['INTERSECTION']):
            found = True
            if 'total tickets' not in cameras[key]['attributes']:
                cameras[key]['attributes']['total tickets'] = 0
                cameras[key]['attributes']['tickets since 12/22/2014'] = 0
                cameras[key]['attributes']['tickets since 3/6/2015'] = 0
                cameras[key]['attributes']['last ticket date'] = line['VIOLATION DATE']
            else:
                cameras[key]['attributes']['total tickets'] += int(line['VIOLATIONS'])
                dt = datetime.strptime(line['VIOLATION DATE'], '%m/%d/%Y')
                if dt >= datetime.strptime('12/22/2014', '%m/%d/%Y'):
                    cameras[key]['attributes']['tickets since 12/22/2014'] += int(line['VIOLATIONS'])
                if dt >= datetime.strptime('3/6/2015', '%m/%d/%Y'):
                    cameras[key]['attributes']['tickets since 3/6/2015'] += int(line['VIOLATIONS'])
    if not found:
        missing_intersections.add(line['INTERSECTION'])
print 'Missing %d intersections' % len(missing_intersections), missing_intersections

import locale
locale.setlocale( locale.LC_ALL, '' )

total = 0
missing_tickets = []
for camera in cameras:
    try:
        total += cameras[camera]['attributes']['total tickets']
    except KeyError:
        missing_tickets.append(camera)

print '%d tickets have been issued since 7/1/2014, raising %s' % (total, locale.currency(total * 100, grouping=True))
print 'The following %d intersections appear to never have issued a ticket in that time: %s' % (
    len(missing_tickets), '; '.join(missing_tickets))

total = 0
low_crash_total = 0
for camera in cameras:
    try:
        total += cameras[camera]['attributes']['tickets since 12/22/2014']
        if cameras[camera]['attributes']['Few crashes']:
            low_crash_total += cameras[camera]['attributes']['tickets since 12/22/2014']
    except KeyError:
        continue

print '%d tickets have been issued at low-crash intersections since 12/22/2014, raising %s' % (
    low_crash_total, locale.currency(low_crash_total * 100, grouping=True))
print '%d tickets have been issued overall since 12/22/2014, raising %s' % (
    total, locale.currency(total * 100, grouping=True))

total = 0
low_crash_total = 0
slated_for_closure_total = 0
for camera in cameras:
    try:
        total += cameras[camera]['attributes']['tickets since 3/6/2015']
        if cameras[camera]['attributes']['Few crashes']:
            low_crash_total += cameras[camera]['attributes']['tickets since 3/6/2015']
        if cameras[camera]['attributes']['To be removed']:
            slated_for_closure_total += cameras[camera]['attributes']['tickets since 3/6/2015']
    except KeyError:
        continue

print '%d tickets have been issued at low-crash intersections since 3/6/2015, raising %s' % (
    low_crash_total, locale.currency(low_crash_total * 100, grouping=True))
print '%d tickets have been issued overall since 3/6/2015, raising %s' % (
    total, locale.currency(total * 100, grouping=True))
print '%d tickets have been issued at cameras that were supposed to be closed since 3/6/2015, raising %s' % (
    slated_for_closure_total, locale.currency(slated_for_closure_total * 100, grouping=True))

from csv import DictWriter
output = []

for camera in cameras:
    data = {
        'intersection': camera,
        'last ticket date': cameras[camera]['attributes'].get('last ticket date', ''),
        'tickets since 7/1/2014': cameras[camera]['attributes'].get('total tickets', 0),
        'revenue since 7/1/2014': cameras[camera]['attributes'].get('total tickets', 0) * 100,
        'tickets since 12/22/2014': cameras[camera]['attributes'].get('tickets since 12/22/2014', 0),
        'revenue since 12/22/2014': cameras[camera]['attributes'].get('tickets since 12/22/2014', 0) * 100,
        'was slated for removal': cameras[camera]['attributes'].get('To be removed', False),
        'had few crashes': cameras[camera]['attributes'].get('Few crashes', False),
        'is currently active': True if 'current' in cameras[camera]['attributes'] else False,
        'latitude': cameras[camera]['attributes'].get('LAT', 0),
        'longitude': cameras[camera]['attributes'].get('LNG', 0)
    }
    output.append(data)

with open('/tmp/red_light_intersections.csv', 'w+') as fh:
    writer = DictWriter(fh, sorted(output[0].keys()))
    writer.writeheader()
    writer.writerows(output)

