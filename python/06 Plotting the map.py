get_ipython().magic('matplotlib inline')

import geopandas as gpd

# These are our three files

# File 1: The filtered list of roads
roads_df = gpd.read_file("singapore-roads-filtered.geojson")
roads_df

# File 2: The list of corrected road names. This is a "bridge" between the name given
# in the GeoJSON file and the actual name used in the classification step.

bridge_df = gpd.pd.read_csv("singapore-roadnames-final-split.csv")
bridge_df.drop(["Unnamed: 0", "has_malay_road_tag"], inplace=True, axis=1)
bridge_df

# And lastly, this file gives us the classification info

classification_df = gpd.pd.read_csv('singapore-roadnames-final-classified.csv')
classification_df.drop(["has_malay_road_tag", "Unnamed: 0", "comment"], inplace=True, axis=1)
classification_df

# First we merge the last two files.

merged_df = bridge_df.merge(classification_df, how="left", on="road_name")
merged_df

# Now we do the second road of merging:

final_df = roads_df.merge(merged_df, how="left", on="name")
final_df

# There are a few NaNs in here as a result of the order in which I did filtering in
# the first notebook, so let's just drop them again

final_df2 = final_df[final_df['classification'].notnull()]
type(final_df2)

# we lost the Geo-ness, convert back to GeoDataFrame
final_df2 = gpd.GeoDataFrame(final_df2)

# Let's save it

final_df2.to_file("singapore-roads-classified.geojson", driver="GeoJSON")

import matplotlib.pyplot as plt
import mplleaflet

final_df2 = gpd.read_file("singapore-roads-classified.geojson")

# GeoPandas' plot function can take a column and colour-code by this column
ax = final_df2.plot(column='classification', colormap='Accent')

mplleaflet.display(fig=ax.figure, crs=final_df2.crs, tiles='cartodb_positron')

mplleaflet.show(fig=ax.figure, crs=final_df2.crs, tiles='cartodb_positron', path='sgmap.html')

# the order is this: a sorted list of values
categories = list(set(final_df2['classification'].values))
categories.sort()

categories

# define the colormap with logical colours (for a Singaporean at least):
# blue for British
# red for Chinese
# etc
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('my cmap', ['blue', 
                                                     'red',
                                                     'gray',
                                                     'yellow',
                                                     'green',
                                                     'purple'])

# pass the custom colormap
ax2 = final_df2.plot(column='classification', colormap=cmap)

mplleaflet.display(fig=ax2.figure, crs=final_df2.crs, tiles='cartodb_positron')

mplleaflet.show(fig=ax2.figure, crs=final_df2.crs, tiles='cartodb_positron', path='sgmap2.html')



from IPython.display import HTML
HTML("<iframe width='100%' height='520' frameborder='0' src='http://michelleful.cartodb.com/viz/b722485c-dbf6-11e4-9a7e-0e0c41326911/embed_map'></iframe>")

