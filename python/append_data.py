import pandas as pd
from arcgis.gis import GIS

from partnerutils.cool_utils import chunk_df
from partnerutils.feature_utils import sdf_from_xyz

# log in to your GIS
gis = GIS(username="mpayson_startups")

# path to data
csv_path = "../test_data/NYC Inspections Geocoded.csv"

x_col = "x"
y_col = "y"

# read csv and construct spatial dataframe
df = pd.read_csv(csv_path)
sdf = sdf_from_xyz(df, x_col, y_col)
len(sdf)

# iterate through chunks to create and append data
lyr = None
for c_df in chunk_df(sdf, 50):
    if not lyr:
        item = c_df.to_featurelayer("MyFeatureService")
        lyr = item.layers[0]
    else:
        # THIS IS THE APPEND DATA PART
        fs = c_df.to_featureset()
        success = lyr.edit_features(adds=fs)
item



