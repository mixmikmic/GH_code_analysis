get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import os
import sqlite3

import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

from odm2api.ODMconnection import dbconnection
import odm2api.ODM2.services.readService as odm2rs

# Assign directory paths and SQLite file name
dpth = os.getcwd()
dbname_sqlite = "ODM2_Example2.sqlite"

sqlite_pth = os.path.join(dpth, os.path.pardir, "data", dbname_sqlite)

try:
    session_factory = dbconnection.createConnection('sqlite', sqlite_pth, 2.0)
    read = odm2rs.ReadODM2(session_factory)
    print("Database connection successful!")
except Exception as e:
    print("Unable to establish connection to the database: ", e)

# Get all of the Variables from the ODM2 database then read the records
# into a Pandas DataFrame to make it easy to view and manipulate
allVars = read.getVariables()

# Get all of the Variables from the ODM2 database then read the records
# into a Pandas DataFrame to make it easy to view and manipulate
allVars = read.getVariables()

variables_df = pd.DataFrame.from_records([vars(variable) for variable in allVars], index='VariableID')
variables_df.head(10)

allPeople = read.getPeople()
pd.DataFrame.from_records([vars(person) for person in allPeople]).head()

# Get all of the SamplingFeatures from the ODM2 database that are Sites
siteFeatures = read.getSamplingFeatures(type='Site')

# Read Sites records into a Pandas DataFrame
# ()"if sf.Latitude" is used only to instantiate/read Site attributes)
df = pd.DataFrame.from_records([vars(sf) for sf in siteFeatures if sf.Latitude])

# Create a GeoPandas GeoDataFrame from Sites DataFrame
ptgeom = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=ptgeom, crs={'init': 'epsg:4326'})
gdf.head(5)

# Number of records (features) in GeoDataFrame
len(gdf)

# A trivial plot, easy to generate thanks to GeoPandas
gdf.plot();

gdf['SiteTypeCV'].value_counts()

gdf["color"] = gdf.apply(lambda feat: 'green' if feat['SiteTypeCV'] == 'Stream' else 'red', axis=1)

sitetype = 'spring'
pd.read_csv("http://vocabulary.odm2.org/api/v1/sitetype/{}/?format=csv".format(sitetype))

c = gdf.unary_union.centroid # GeoPandas heavy lifting
m = folium.Map(location=[c.y, c.x], tiles='CartoDB positron', zoom_start=11)

marker_cluster = folium.MarkerCluster().add_to(m)
for idx, feature in gdf.iterrows():
    folium.Marker(location=[feature.geometry.y, feature.geometry.x], 
                  icon=folium.Icon(color=feature['color']),
                  popup="{0} ({1}): {2}".format(
                      feature['SamplingFeatureCode'], feature['SiteTypeCV'], feature['SamplingFeatureName'])
                 ).add_to(marker_cluster)

    
# Done with setup. Time to render the map
m

# Get the SamplingFeature object for a particular SamplingFeature by passing its SamplingFeatureCode
sf = read.getSamplingFeatures(codes=['RB_1300E'])[0]
type(sf)

# Simple way to examine the content (properties) of a Python object, as if it were a dictionary
vars(sf)

print("\n------------ Foreign Key Example --------- \n")
try:
    # Call getResults, but return only the first Result
    firstResult = read.getResults()[0]
    print("The FeatureAction object for the Result is: ", firstResult.FeatureActionObj)
    print("The Action object for the Result is: ", firstResult.FeatureActionObj.ActionObj)
    
    # Or, print some of those attributes in a more human readable form:
    print("\nThe following are some of the attributes for the Action that created the Result: " +
          "\nActionTypeCV: " + firstResult.FeatureActionObj.ActionObj.ActionTypeCV + 
          "\nActionDescription: " + str(firstResult.FeatureActionObj.ActionObj.ActionDescription) + 
          "\nBeginDateTime: " + str(firstResult.FeatureActionObj.ActionObj.BeginDateTime) + 
          "\nEndDateTime: " + str(firstResult.FeatureActionObj.ActionObj.EndDateTime) + 
          "\nMethodName: " + firstResult.FeatureActionObj.ActionObj.MethodObj.MethodName + 
          "\nMethodDescription: " + firstResult.FeatureActionObj.ActionObj.MethodObj.MethodDescription)
except Exception as e:
    print("Unable to demo Foreign Key Example: ", e)

# Get a particular Result
print("\n------- Example of Retrieving Attributes of a Result -------")
try:
    firstResult = read.getResults()[0]
    print(
        "The following are some of the attributes for the Result retrieved: \n" +
        "ResultID: " + str(firstResult.ResultID) + "\n" +
        "ResultTypeCV: " + firstResult.ResultTypeCV + "\n" +
        "ValueCount: " + str(firstResult.ValueCount) + "\n" +
        # Get the ProcessingLevel from the Result's ProcessingLevel object
        "ProcessingLevel: " + firstResult.ProcessingLevelObj.Definition + "\n" +
        "SampledMedium: " + firstResult.SampledMediumCV + "\n" +
        # Get the Variable information from the Result's Variable object
        "Variable: " + firstResult.VariableObj.VariableCode + ": " + firstResult.VariableObj.VariableNameCV + "\n" +
        # Get the Units information from the Result's Units object
        "Units: " + firstResult.UnitsObj.UnitsName + "\n" +
        # Get the Specimen information by drilling down into the result object
        "SamplingFeatureID: " + str(firstResult.FeatureActionObj.SamplingFeatureObj.SamplingFeatureID) + "\n" +
        "SamplingFeatureCode: " + firstResult.FeatureActionObj.SamplingFeatureObj.SamplingFeatureCode)
except Exception as e:
    print("Unable to demo example of retrieving Attributes of a Result: ", e)

# Pass the Sampling Feature ID of the specimen, and the relationship type
relatedSite = read.getRelatedSamplingFeatures(sfid=26, relationshiptype='Was Collected at')[0]

vars(relatedSite)

siteID = 1  # Red Butte Creek at 1300 E (obtained from the getRelatedSamplingFeatures query)

# Get a list of Results at a particular Site, for a particular Variable, and of type "Measurement"
v = variables_df[variables_df['VariableCode'] == 'TP']
variableID = v.index[0]

results = read.getResults(siteid=siteID, variableid=variableID, type="Measurement")
# Get the list of ResultIDs so I can retrieve the data values associated with all of the results
resultIDList = [x.ResultID for x in results]
len(resultIDList)

# Get all of the data values for the Results in the list created above
# Call getResultValues, which returns a Pandas Data Frame with the data
resultValues = read.getResultValues(resultids=resultIDList)
resultValues.head()

# Plot the time sequence of Measurement Result Values 
resultValues.plot(x='valuedatetime', y='datavalue', title=relatedSite.SamplingFeatureName,
                  kind='line', use_index=True, linestyle='solid', style='o')
ax = plt.gca()
ax.set_ylabel("{0} ({1})".format(results[0].VariableObj.VariableNameCV, 
                                 results[0].UnitsObj.UnitsAbbreviation))
ax.set_xlabel('Date/Time')
ax.grid(True)
ax.legend().set_visible(False)

def get_results_and_values(siteid, variablecode):
    v = variables_df[variables_df['VariableCode'] == variablecode]
    variableID = v.index[0]
    
    results = read.getResults(siteid=siteid, variableid=variableID, type="Measurement")
    resultIDList = [x.ResultID for x in results]
    resultValues = read.getResultValues(resultids=resultIDList)
    
    return resultValues, results

# Plot figure and axis set up (just *one* subplot, actually)
f, ax = plt.subplots(1, figsize=(13, 6))

# First plot (left axis)
VariableCode = 'TP'
resultValues_TP, results_TP = get_results_and_values(siteID, VariableCode)
resultValues_TP.plot(x='valuedatetime', y='datavalue', label=VariableCode, 
                     style='o-', kind='line', ax=ax)
ax.set_ylabel("{0}: {1} ({2})".format(VariableCode, results_TP[0].VariableObj.VariableNameCV, 
                                      results_TP[0].UnitsObj.UnitsAbbreviation))

# Second plot (right axis)
VariableCode = 'TN'
resultValues_TN, results_TN = get_results_and_values(siteID, VariableCode)
resultValues_TN.plot(x='valuedatetime', y='datavalue', label=VariableCode, 
                     style='^-', kind='line', ax=ax,
                     secondary_y=True)
ax.right_ax.set_ylabel("{0}: {1} ({2})".format(VariableCode, results_TN[0].VariableObj.VariableNameCV, 
                                               results_TN[0].UnitsObj.UnitsAbbreviation))

# Tweak the figure
ax.legend(loc=2)
ax.right_ax.legend(loc=1)

ax.grid(True)
ax.set_xlabel('')
ax.set_title(relatedSite.SamplingFeatureName);

print("TP METHOD:  {0} ({1})".format(results_TP[0].FeatureActionObj.ActionObj.MethodObj.MethodName,
                                     results_TP[0].FeatureActionObj.ActionObj.MethodObj.MethodDescription))

print("TN METHOD:  {0} ({1})".format(results_TN[0].FeatureActionObj.ActionObj.MethodObj.MethodName,
                                     results_TN[0].FeatureActionObj.ActionObj.MethodObj.MethodDescription))

