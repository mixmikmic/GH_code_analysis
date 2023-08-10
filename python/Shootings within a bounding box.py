import os
import requests

def get_data(table):
    r = requests.get('%stable/json/%s' % (os.environ['NEWSROOMDB_URL'], table))
    return r.json()

shootings = get_data('shootings')

from shapely.geometry import Point, Polygon

northeast = [-87.676145, 41.777527]
northwest = [-87.690842, 41.777228]
southeast = [-87.675973, 41.766610]
southwest = [-87.690651, 41.768241]
bounds = Polygon([northwest, northeast, southeast, southwest])

for row in shootings:
    if not row['Geocode Override']:
        continue
    points = row['Geocode Override'][1:-1].split(',')
    if len(points) != 2:
        continue
    point = Point(float(points[1]), float(points[0]))
    if bounds.contains(point):
        print row['Date'], row['Age'], row['Sex'], '\n'
        

