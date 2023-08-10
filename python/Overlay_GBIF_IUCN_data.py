get_ipython().magic('matplotlib inline')
import logging
root = logging.getLogger()
root.addHandler(logging.StreamHandler())

from iSDM.species import GBIFSpecies
gbif_species = GBIFSpecies(name_species="Graptemys oculifera")
gbif_species.find_species_occurrences().head()

gbif_species.save_data()

gbif_species.plot_species_occurrence()

gbif_species.geometrize(dropna=True) # converts the lat/lon columns into a geometrical Point, for each record

gbif_species.get_data().head() # notice the last 'geometry' column added now

gbif_species.get_data().shape # there are 40 records left (containing lat/lon) at this point.

from iSDM.species import IUCNSpecies
iucn_species = IUCNSpecies(name_species='Graptemys oculifera')
iucn_species.load_shapefile('../data/FW_TURTLES/FW_TURTLES.shp')

iucn_species.find_species_occurrences()  # IUCN datasets have a 'geometry' column

iucn_species.plot_species_occurrence() # the rangemap seems to be around the same area as the point-records

backup_gbif_species = gbif_species.get_data().copy()

gbif_species.overlay(iucn_species)

gbif_species.get_data().shape # 39 records, so one unlucky observation falls outside the rangemap

gbif_species.plot_species_occurrence()

gbif_species_geometry = gbif_species.get_data().geometry
iucn_species_geometry = iucn_species.get_data().geometry

from geopandas import GeoSeries
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
combined_geometries = GeoSeries(gbif_species_geometry.append(iucn_species_geometry))
combined_geometries.plot()

gbif_species_geometry.head()

iucn_species_geometry.head()

combined_geometries = GeoSeries(backup_gbif_species.geometry.append(iucn_species_geometry))
plt.figure(figsize=(15,15))
combined_geometries.plot()





