#import the package
import geopandas as gpd

#enable plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#read in the HUC12.shp feature class
gdf = gpd.read_file('../Data/HUC12.shp')

#How many features in the dataset?
len(gdf)

#show the data types in this dataset
gdf.dtypes

#examine the attributes for the first feature
gdf.iloc[0]

#show the first 5 values in the geometry field
gdf['geometry'][0:5]

#show just the first value
gdf['geometry'][0]

#Plotting - http://geopandas.org/mapping.html
gdf.plot(column='HUC_8',
         cmap='Paired',
         categorical=True,
         figsize=(14,18)
        );

#Dissolving
dfHUC8 = gdf.dissolve(by='HUC_8',aggfunc='sum')
dfHUC8.dtypes

dfHUC8.plot(column='ACRES',
            scheme='quantiles',        
            figsize=(14,18));

