import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
root = logging.getLogger()
root.addHandler(logging.StreamHandler())
get_ipython().magic('matplotlib inline')

# download from Google Drive: https://drive.google.com/open?id=0B9cazFzBtPuCOFNiUHYwcVFVODQ
# Representative example with multiple polygons in the shapefile, and a lot of point-records (also outside rangemaps)
from iSDM.species import IUCNSpecies
salmo_trutta = IUCNSpecies(name_species='Salmo trutta')
salmo_trutta.load_shapefile("../data/fish/selection/salmo_trutta")

rasterized = salmo_trutta.rasterize(raster_file="./salmo_trutta_full.tif", pixel_size=0.5, all_touched=True)

plt.figure(figsize=(25,20))
plt.imshow(rasterized, cmap="hot", interpolation="none")

from iSDM.environment import RasterEnvironmentalLayer
biomes_adf = RasterEnvironmentalLayer(file_path="../data/rebioms/w001001.adf", name_layer="Biomes")
biomes_adf.load_data()

biomes_adf.plot()

from iSDM.environment import ContinentsLayer
from iSDM.environment import Source
continents = ContinentsLayer(file_path="../data/continents/continent.shp", source=Source.ARCGIS)
continents.load_data()
fig, ax = plt.subplots(1,1, figsize=(30,20))
continents.data_full.plot(column="continent", colormap="hsv")

continents_rasters = continents.rasterize(raster_file="../data/continents/continents_raster.tif", pixel_size=0.5, all_touched=True)

continents_rasters.shape # stacked raster with 8 bands, one for each continent.

selected_layers, pseudo_absences = biomes_adf.sample_pseudo_absences(species_raster_data=rasterized, continents_raster_data=continents_rasters, number_of_pseudopoints=1000)

plt.figure(figsize=(25,20))
plt.imshow(selected_layers, cmap="hot", interpolation="none")

plt.figure(figsize=(25,20))
plt.imshow(pseudo_absences, cmap="hot", interpolation="none")

all_coordinates = biomes_adf.pixel_to_world_coordinates(raster_data=np.zeros_like(rasterized), filter_no_data_value=False)

all_coordinates

base_dataframe = pd.DataFrame([all_coordinates[0], all_coordinates[1]]).T
base_dataframe.columns=['decimallatitude', 'decimallongitude']

base_dataframe.set_index(['decimallatitude', 'decimallongitude'], inplace=True, drop=True)

base_dataframe.head()

base_dataframe.tail()

presence_coordinates = salmo_trutta.pixel_to_world_coordinates()

presence_coordinates

presences_dataframe = pd.DataFrame([presence_coordinates[0], presence_coordinates[1]]).T
presences_dataframe.columns=['decimallatitude', 'decimallongitude']
presences_dataframe[salmo_trutta.name_species] = 1 # fill presences with 1's
presences_dataframe.set_index(['decimallatitude', 'decimallongitude'], inplace=True, drop=True)

presences_dataframe.head()

presences_dataframe.tail()

pseudo_absence_coordinates = biomes_adf.pixel_to_world_coordinates(raster_data=pseudo_absences)

pseudo_absences_dataframe = pd.DataFrame([pseudo_absence_coordinates[0], pseudo_absence_coordinates[1]]).T
pseudo_absences_dataframe.columns=['decimallatitude', 'decimallongitude']
pseudo_absences_dataframe[salmo_trutta.name_species] = 0
pseudo_absences_dataframe.set_index(['decimallatitude', 'decimallongitude'], inplace=True, drop=True)

pseudo_absences_dataframe.head()

pseudo_absences_dataframe.tail()

from iSDM.environment import ClimateLayer
water_min_layer =  ClimateLayer(file_path="../data/watertemp/min_wt_2000.tif") 
water_min_reader = water_min_layer.load_data()
# HERE: should we ignore cells with no-data values for temperature? They are set to a really big negative number
# for now we keep them, otherwise could be NaN
water_min_coordinates = water_min_layer.pixel_to_world_coordinates(filter_no_data_value=False)

water_min_coordinates

mintemp_dataframe = pd.DataFrame([water_min_coordinates[0], water_min_coordinates[1]]).T
mintemp_dataframe.columns=['decimallatitude', 'decimallongitude']
water_min_matrix = water_min_reader.read(1)
mintemp_dataframe['MinT'] = water_min_matrix.reshape(np.product(water_min_matrix.shape))
mintemp_dataframe.set_index(['decimallatitude', 'decimallongitude'], inplace=True, drop=True)
mintemp_dataframe.head()

mintemp_dataframe.tail()

water_max_layer =  ClimateLayer(file_path="../data/watertemp/max_wt_2000.tif") 
water_max_reader = water_max_layer.load_data()
# HERE: should we ignore cells with no-data values for temperature? They are set to a really big negative number
# for now we keep them, otherwise could be NaN
water_max_coordinates = water_max_layer.pixel_to_world_coordinates(filter_no_data_value=False)

maxtemp_dataframe = pd.DataFrame([water_max_coordinates[0], water_max_coordinates[1]]).T
maxtemp_dataframe.columns=['decimallatitude', 'decimallongitude']
water_max_matrix = water_max_reader.read(1)
maxtemp_dataframe['MaxT'] = water_max_matrix.reshape(np.product(water_max_matrix.shape))
maxtemp_dataframe.set_index(['decimallatitude', 'decimallongitude'], inplace=True, drop=True)
maxtemp_dataframe.head()

maxtemp_dataframe.tail()

water_mean_layer =  ClimateLayer(file_path="../data/watertemp/mean_wt_2000.tif") 
water_mean_reader = water_mean_layer.load_data()
# HERE: should we ignore cells with no-data values for temperature? They are set to a really big negative number
# for now we keep them, otherwise could be NaN
water_mean_coordinates = water_mean_layer.pixel_to_world_coordinates(filter_no_data_value=False)

meantemp_dataframe = pd.DataFrame([water_mean_coordinates[0], water_mean_coordinates[1]]).T
meantemp_dataframe.columns=['decimallatitude', 'decimallongitude']
water_mean_matrix = water_mean_reader.read(1)
meantemp_dataframe['MeanT'] = water_mean_matrix.reshape(np.product(water_mean_matrix.shape))
meantemp_dataframe.set_index(['decimallatitude', 'decimallongitude'], inplace=True, drop=True)
meantemp_dataframe.head()

meantemp_dataframe.tail()

# merge base with presences
merged = base_dataframe.combine_first(presences_dataframe)

merged.head()

merged.tail()

# merge based+presences with pseudo-absences
# merged2 = pd.merge(merged1, pseudo_absences_dataframe, on=["decimallatitude", "decimallongitude", salmo_trutta.name_species], how="outer")

merged = merged.combine_first(pseudo_absences_dataframe)

merged.head()

merged.tail()

# merge base+presences+pseudo-absences with min temperature
#merged3 = pd.merge(merged2, mintemp_dataframe, on=["decimallatitude", "decimallongitude"], how="outer")

merged = merged.combine_first(mintemp_dataframe)

merged.head()

merged.tail()

# merged4 = pd.merge(merged3, maxtemp_dataframe, on=["decimallatitude", "decimallongitude"], how="outer")
merged = merged.combine_first(maxtemp_dataframe)

merged.head()

merged.tail()

# merged5 = pd.merge(merged4, meantemp_dataframe, on=["decimallatitude", "decimallongitude"], how="outer")
merged = merged.combine_first(meantemp_dataframe)

merged.tail()

merged.to_csv("../data/fish/selection/salmo_trutta_again.csv")

merged[merged['Salmo trutta']==0].shape[0] # should be equal to number of pseudo absences below

pseudo_absence_coordinates[0].shape[0]

merged[merged['Salmo trutta']==1].shape[0]  # should be equal to number of presences below

presence_coordinates[0].shape[0]

merged[merged['Salmo trutta'].isnull()].shape[0] # all that's left

360 * 720 == merged[merged['Salmo trutta']==0].shape[0] + merged[merged['Salmo trutta']==1].shape[0] + merged[merged['Salmo trutta'].isnull()].shape[0]

# == all pixels in 360 x 720 matrix

merged[merged['Salmo trutta']==0.0]

pseudo_absences_dataframe



