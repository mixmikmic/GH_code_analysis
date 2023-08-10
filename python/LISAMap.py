get_ipython().magic('matplotlib inline')

from arcgis.gis import GIS
from arcgis.features import SpatialDataFrame
from arcgis.features import FeatureCollection
import numpy as np
import pysal as ps
import pandas as pd

fc = ps.examples.get_path('NAT.shp')

from arcgis.features import FeatureSet
sdf = SpatialDataFrame.from_featureclass(fc)
fs = FeatureSet.from_dict(sdf.__feature_set__)
collection = FeatureCollection.from_featureset(fs)

gis = GIS()
m = gis.map('United States')

m

m.add_layer(collection)

w = ps.weights.Queen.from_dataframe(sdf, geom_col='SHAPE')
hr90 = sdf['HR90']
lisa = ps.Moran_Local(hr90, w, permutations=9999)

lisa_lbls = {1: 'HH', 
             2: 'LH', 
             3: 'LL', 
             4: 'HL', 
             0: 'Non-significant'}
sign = lisa.p_sim < 0.05
quadS = lisa.q * sign
labels = pd.Series(quadS).map(lisa_lbls).values
sdf['LISALABELS'] = labels
sdf['LISAVALUES'] = pd.Series(quadS).values

sdf.LISALABELS.value_counts().plot('bar')

item = gis.content.search("id: 192a1923c2274af890645c5465184d35")[0]
item

m = gis.map(item)
m

