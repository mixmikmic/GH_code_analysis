get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

import seaborn as sns

gdf = gpd.read_file("C:/WorkSpace/pacsafeapp/data/Nukualofa/to_buildings.shp")

gdf.info()

gdf.head()

wall_frame = gdf.groupby('B_FRAME1')
wall_material = gdf.groupby('WALL_MAT1')
roof_material = gdf.groupby('ROOF_MAT_1')
foundations = gdf.groupby('FOUND1')
floor_height = gdf.groupby('F_MINHT')
nrecords = len(gdf)

pwallframe = 100 * np.count_nonzero(gdf['B_FRAME1'].notnull())/float(nrecords)
pwall_material = 100 * np.count_nonzero(gdf['WALL_MAT1'].notnull())/float(nrecords)
proof_material = 100 * np.count_nonzero(gdf['ROOF_MAT_1'].notnull())/float(nrecords)
pfoundations = 100 * np.count_nonzero(gdf['FOUND1'].notnull())/float(nrecords)
pfloor_height = 100 * np.count_nonzero(gdf['F_MINHT'].notnull())/float(nrecords)

print("Percentage of complete records")
print("------------------------------")
print("Wall frame:    {0:.2f}%".format(pwallframe))
print("Wall material: {0:.2f}%".format(pwall_material))
print("Roof material: {0:.2f}%".format(proof_material))
print("Foundation:    {0:.2f}%".format(pfoundations))
print("Floor height   {0:.2f}%".format(pfloor_height))

100 * wall_frame.count()['AGE']/nrecords

100 * wall_material.count()['AGE']/nrecords

100 * roof_material.count()['AGE']/nrecords

100 * foundations.count()['AGE']/nrecords

grouped = gdf.groupby(['USE_GRP', 'B_FRAME1'])
100 * grouped.count()['AGE']/len(gdf)

def autolabel(rects, rotation='horizontal'):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if np.isnan(height):
            height = 0
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom', rotation=rotation, fontsize='small')

fig, ax = plt.subplots(1, 1, figsize=(16,8))
ax = sns.countplot(x='USE_GRP', data=gdf, palette='RdBu', hue='B_FRAME1')
autolabel(ax.patches, rotation='vertical')
ax.legend(loc=1, title="Wall frame type")
ax.set_xlabel('Building use group')
ax.set_title("Building stock - Tonga")
#labels = ax.get_xticklabels()
#ax.set_xticklabels(labels,rotation='vertical')

gdf_valid = gdf[gdf['B_FRAME1'].notnull()]

fig, ax = plt.subplots(1, 1, figsize=(6, 16))

villages = gpd.read_file("R:/Pacific/data/external/pcrafi/TO/to_village.shp")
base = villages.plot(ax=ax, cmap='Blues')
gdf.plot(ax=base, color='k')
gdf_valid.plot(ax=base, color='r')



