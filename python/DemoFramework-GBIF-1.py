from iSDM.species import GBIFSpecies

my_species = GBIFSpecies(name_species="Etheostoma_blennioides")

my_species.name_species

get_ipython().magic('matplotlib inline')
import logging
root = logging.getLogger()
root.addHandler(logging.StreamHandler())

my_species.find_species_occurrences().head()

my_species.ID # taxonkey derived from GBIF. It's a sort of unique ID per species

my_species.save_data()

my_species.source.name

my_species.plot_species_occurrence()

polygonized_species = my_species.polygonize()

my_species.overlay(polygonized_species.geometry)
my_species.data_full.shape

polygonized_species.geometry = polygonized_species.geometry[7:]

polygonized_species.dropna()

polygonized_species.dropna(inplace=True)

my_species.overlay(polygonized_species.geometry)
my_species.data_full.shape

my_species.plot_species_occurrence()

data = my_species.load_data("./Etheostoma_blennioides2382397.pkl") # or just load existing data into Species object

data.columns # all the columns available per observation

data.head()

data['country'].unique().tolist()

data.shape # there are 7226 observations, 138 parameters per observation

data['vernacularname'].unique().tolist() # self-explanatory

data['decimallatitude'].tail(10)

import numpy as np
data_cleaned = data.dropna(subset = ['decimallatitude', 'decimallongitude']) # drop records where data not available

data_cleaned.shape # less occurrence records now: 5226

data_cleaned['basisofrecord'].unique()

# this many records with no decimalLatitude and decimalLongitude
import numpy as np
data[data['decimallatitude'].isnull() & data['decimallongitude'].isnull()].size

data[data['decimallatitude'].isnull() & 
     data['decimallongitude'].isnull() & 
     data['locality'].isnull() & 
     data['verbatimlocality'].isnull()]

data_cleaned[['dateidentified', 'day', 'month', 'year']].head()

data_selected = data_cleaned[data_cleaned['year']>2010][['decimallatitude','decimallongitude', 'rightsholder', 'datasetname']]

data_selected[~data_selected.datasetname.isnull()].head(10)

my_species.set_data(data_selected) # update the object "my_species" to contain the filtered data

my_species.save_data(file_name="updated_dataset.pkl")

my_species.plot_species_occurrence()

my_species.get_data().shape # there are 119 records now

csv_data = my_species.load_csv('../data/GBIF.csv') 

csv_data.head() # let's peak into the data

csv_data['specieskey'].unique()

my_species.save_data() # by default this 'speciesKey' is used. Alternative name can be provided

csv_data.columns.size # csv data for some reason a lot less columns

data.columns.size # data from using GBIF API directly

list(set(data.columns.tolist()) - set(csv_data.columns.tolist())) # hmm, 'decimalLatitude' vs 'decimallatitude'

list(set(csv_data.columns.tolist()) - set(data.columns.tolist())) # hmm, not many

geometrized_species =  my_species.polygonize()  # returns a geopandas dataframe with a geometry column.

geometrized_species

geometrized_species.plot()  # each isolated polygon is a separate record (do we want that or?)

# we can tweak the parameters for the polygonize function
geometrized_species = my_species.polygonize(buffer_distance=0.2, simplify_tolerance=0.02)
geometrized_species.plot()

my_species.get_data().shape

# with_envelope means "pixelized" (envelope around each buffer region)
geometrized_species = my_species.polygonize(buffer_distance=0.3, simplify_tolerance=0.03, with_envelope=True)
geometrized_species.plot()

from shapely.geometry import Point, Polygon

# say we want to crop to this polygon area only
overlay_polygon = Polygon(((-100,30), (-100, 50), (-70, 50),(-70, 30)))

# Beware, this overwrites the original my_species data ("data_full" field)
my_species.data_full = my_species.data_full[my_species.data_full.geometry.within(overlay_polygon)]

my_species.polygonize().plot()

my_species.polygonize(buffer_distance=0.5, simplify_tolerance=0.05).plot() # more fine-grained

my_species.polygonize(buffer_distance=0.3, simplify_tolerance=0.03).plot()  # etc

my_species.polygonize(buffer_distance=0.3, simplify_tolerance=0.03, with_envelope=True).plot() # with_envelope means pixelized

# we can further simplify with a "convex hull" around each polygon
my_species.polygonize().geometry.convex_hull.plot()

polygonized_species = my_species.polygonize()

polygonized_species

polygonized_species.plot()

# We can make a union of all polygons into one "multipolygon" (Do we need this? I can make a wrapper if needed)
import shapely.ops
my_multipolygon = shapely.ops.cascaded_union(polygonized_species.geometry.tolist())
my_multipolygon

from geopandas import GeoDataFrame, GeoSeries
new_series = GeoSeries(shapely.ops.cascaded_union(polygonized_species.geometry.tolist()))
new_series

new_series.plot()

new_series.convex_hull.plot()

my_species.data_full.geometry.total_bounds

my_species.data_full.geometry.bounds.minx.min()

my_species.data_full.geometry.bounds.miny.min()

my_species.data_full.geometry.bounds.maxx.max()

my_species.data_full.geometry.bounds.maxy.max()



