import logging
root = logging.getLogger()
root.addHandler(logging.StreamHandler())
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from iSDM.species import IUCNSpecies
turtles = IUCNSpecies(name_species='Pelusios rhodesianus')
turtles.load_shapefile('../data/FW_TURTLES/FW_TURTLES.shp')

turtles.find_species_occurrences()

type(turtles)

turtles.plot_species_occurrence()

turtles.save_shapefile("../data/filtered_turtles_again.shp")

from iSDM.environment import ClimateLayer
worldclim_max_june =  ClimateLayer(file_path="../data/tmax1/tmax6.bil") 
worldclim_max_june.load_data()

worldclim_data = worldclim_max_june.read(1) # read band_1
# get the min (ignoring nodata=-9999)/max values to adjust the spectrum.
vmin = worldclim_data[worldclim_data!=worldclim_max_june.metadata["nodata"]].min()
vmax = worldclim_data.max()

fig, ax = plt.subplots(figsize=(18, 18))
ax.imshow(worldclim_max_june.read(1), cmap="coolwarm", interpolation="none", vmin=vmin, vmax=vmax)

worldclim_max_june.overlay(turtles)
# if i don't limit the plot to vmin/vmax, the image is almost white, because low values are really low. 
# So we need scaling of the colormap
vmin = worldclim_max_june.masked_data.data[worldclim_max_june.masked_data.data!=worldclim_max_june.metadata['nodata']].min()
vmax=worldclim_max_june.masked_data.data.max()

fig, ax = plt.subplots(figsize=(18, 18))
ax.imshow(worldclim_max_june.masked_data, cmap="coolwarm", interpolation="none", vmin=vmin, vmax=vmax) # vmin/vmax for better contrast?

worldclim_max_june.masked_data # there are .data and .mask arrays

worldclim_max_june.masked_data.mask

worldclim_max_june.masked_data.data

# convert to lower resolution?
worldclim_max_june.reproject(destination_file="../data/reprojected.adf", resolution=1, driver='AIG')

# load the lower resolution file? or should reprojecting "overwrite" the original data
worldclim_max_june.load_data("../data/reprojected.adf")

fig, ax = plt.subplots(figsize=(18, 18))
vmin = worldclim_max_june.masked_data.data[worldclim_max_june.masked_data.data!=worldclim_max_june.metadata['nodata']].min()
vmax=worldclim_max_june.masked_data.data.max()
ax.imshow(worldclim_max_june.read(1), cmap="coolwarm", interpolation="none", vmin=vmin, vmax=vmax)

worldclim_max_june.overlay(turtles)
# if i don't limit the plot to vmin/vmax, the image is almost white, because low values are really low. 
# So we need scaling of the colormap
vmin = worldclim_max_june.masked_data.data[worldclim_max_june.masked_data.data!=worldclim_max_june.metadata['nodata']].min()
vmax = worldclim_max_june.masked_data.data.max()

fig, ax = plt.subplots(figsize=(18, 18))
ax.imshow(worldclim_max_june.masked_data, cmap="coolwarm", interpolation="none", vmin=vmin, vmax=vmax) # vmin/vmax for better contrast?



