"""
import `arcpy` and other modules, and make some environment settings here. 
"""
import arcpy
import os
from pprint import pprint

arcpy.env.workspace = '.\\WorkingWithFeatureData.gdb'
arcpy.env.scratchWorkspace = '.\\scratch.gdb'
arcpy.env.overwriteOutput = True

# https://pro.arcgis.com/en/pro-app/arcpy/functions/listfields.htm
# look at the first 5 fields in the data

accidents = os.path.join(arcpy.env.workspace, 'accidents')
fields = arcpy.ListFields(accidents)
pprint([(f.name, f.type, f.aliasName) for f in fields[:5]])

"""
https://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm

- Checking the data using arcpy.da.SearchCursor.
- "Bad" pattern to use '*' all field names.
"""

print('####### Using for loop #######')
with arcpy.da.SearchCursor(accidents, '*') as cursor:  # cursor is an iterator of tuples
    for row in cursor:
        print(row)  # print the values for all columns
        print(row[:2])  # print the values for first two columns
        break

print('\n####### Using next() #######')
with arcpy.da.SearchCursor(accidents, '*') as cursor:
    # fetch the next row in cursor
    # In python 2.7, cursor.next()
    print(next(cursor))

"""
https://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm

- Looking at particular fields in arcpy.da.SearchCursor
- "Good" pattern to specifiy the fields of interests
"""

print('{:>6}   {:>6}'.format('OBJECTID', 'LIGHT_CODE'))

with arcpy.da.SearchCursor(accidents, ['OBJECTID', 'LIGHT_CODE']) as cursor:
    for row in cursor:
        print('{:>6}   {:>6}'.format(row[0], row[1]))

"""
https://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm

- Looking at particular fields in arcpy.da.SearchCursor, with where_clause
- "Good" pattern to specifiy the fields of interests
"""

print('{:>6}   {:>6}'.format('OBJECTID', 'LIGHT_CODE'))

where_clause = "LIGHT_CODE in (0, 88, 99)"
with arcpy.da.SearchCursor(accidents, ['OBJECTID', 'LIGHT_CODE'],
                           where_clause) as cursor:
    for row in cursor:
        print('{:>6}   {:>6}'.format(row[0], row[1]))

"""
https://pro.arcgis.com/en/pro-app/arcpy/data-access/editor.htm
https://pro.arcgis.com/en/pro-app/arcpy/data-access/updatecursor-class.htm

Use `UpdateCursor` to remove records with invalid light code.

Also use `arcpy.da.Editor` which will protect us from "bad" editing.
It groups edits into atomic: either
- all operations are successful and edits are committed, or
- no change will be committed if any operation failed.
"""

where_clause = "LIGHT_CODE in (0, 88, 99)"

# Editor class groups edits into atomic.
# If an error occurs before all edits are completed, the transaction can be rolled back.

with arcpy.da.Editor(arcpy.env.workspace) as editor:
    with arcpy.da.UpdateCursor(accidents, 'LIGHT_CODE', where_clause) as cursor:
        for row in cursor:
            cursor.deleteRow()
            # cursor.UpdateRow(<tuples of updated values>)

# https://pro.arcgis.com/en/pro-app/arcpy/data-access/walk.htm
# use arcpy.da.Walk to traverse directories

for dirpath, dirnames, filenames in arcpy.da.Walk(arcpy.env.scratchWorkspace,
                                                  datatype='FeatureClass',
                                                  type='Polyline'):
        print(dirpath)
        print(dirnames)
        pprint(filenames)

# create a new feature class to store all the road data
# template_fc is used to maintain the same schema

template_fc = os.path.join(arcpy.env.scratchWorkspace, 'county_1')
roads = arcpy.management.CreateFeatureclass(arcpy.env.workspace, 'roads',
                                            geometry_type='Polyline',
                                            template=template_fc)[0]

# roads is currently empty
with arcpy.da.SearchCursor(roads, '*') as cursor:
    print(next(cursor))

"""
https://pro.arcgis.com/en/pro-app/arcpy/data-access/walk.htm
https://pro.arcgis.com/en/pro-app/arcpy/data-access/insertcursor-class.htm

Read every road segment data (including geometry), and insert into new feature class
"""

# get the path of all road segments
for dirpath, dirnames, filenames in arcpy.da.Walk(arcpy.env.scratchWorkspace):
    road_segments = [os.path.join(dirpath, filename) for filename in filenames]

# create geometries requires 'Shape@' token
fields = [f.name for f in arcpy.ListFields(roads)] + ['Shape@']

with arcpy.da.Editor(arcpy.env.workspace) as editor:
    with arcpy.da.InsertCursor(roads, fields) as roads_cursor:  # create cursor for inserting

        # get each row from road segments and insert it into roads
        for road_segment in road_segments:
            with arcpy.da.SearchCursor(road_segment, fields) as segment_cursor:
                for row in segment_cursor:
                    roads_cursor.insertRow(row)

# check the contents in roads
with arcpy.da.SearchCursor(roads, fields) as cursor:
    print(next(cursor))

"""
The following code is to transfer the distribution of crashes onto the road data
via the following tools:
- create Hotspot layer of accident data
- join the number of accidents in each hotspot cell onto road data
"""

hotspot = arcpy.stats.OptimizedHotSpotAnalysis(accidents, 'accidentsHotSpot')[0]
roadWithAccidentsCount = arcpy.analysis.SpatialJoin(roads, hotspot,
                                                'roadWithAccidentsCount',
                                                match_option='INTERSECT')[0]

# https://pro.arcgis.com/en/pro-app/arcpy/data-access/featureclasstonumpyarray.htm
# turn feature class to numpy array using arcpy.da.FeatureClassToNumPyArray

fields = ['Join_Count', 'RURAL_URBA', 'AADT']
road_np = arcpy.da.FeatureClassToNumPyArray(roadWithAccidentsCount,
                                            fields, skip_nulls=True)

print(type(road_np))
print(road_np)

""" read numpy array as pandas dataframe """
import pandas as pd

# read road_np as pandas DataFrame
df = pd.DataFrame(road_np)

# rename the column names
column_labels = {'Join_Count': 'Crash Counts',
                 'RURAL_URBA': 'Rural or Urban',
                 'AADT': 'Average Daily Traffic'}
df = df.rename(columns=column_labels)
df.head()

get_ipython().run_line_magic('matplotlib', 'inline')

# looking at the relationships between number of accidents and Rural/Urban roads
ax = df.boxplot(column='Crash Counts', by='Rural or Urban')
ax.set_ylabel('Crash Counts')
ax.set_title('')

# looking at the relationship between number of crashes and average daily traffic
df.plot.scatter(x='Average Daily Traffic', y='Crash Counts')

