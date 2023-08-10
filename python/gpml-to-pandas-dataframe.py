import pygplates
import pandas as pd
import numpy as np


def feature_collection_to_dataframe(file):
    # function to read in any gplates-compatible feature collection and 
    # place it into a pandas dataframe

    feature_collection = pygplates.FeatureCollection(file)

    DataFrameTemplate = ['lon','lat']

    # Get attribute (other than coordinate) names from first feature
    for feature in feature_collection: 
        for attribute in feature.get_shapefile_attributes():
            DataFrameTemplate.append(attribute) 
        break

    res = []
    for feature in feature_collection:
        tmp = []
        tmp.append(feature.get_geometry().to_lat_lon()[1])
        tmp.append(feature.get_geometry().to_lat_lon()[0])
        for attribute in feature.get_shapefile_attributes():
            tmp.append(feature.get_shapefile_attribute(attribute))
        res.append(tmp)

    df = pd.DataFrame(res,columns=DataFrameTemplate)

    return df





