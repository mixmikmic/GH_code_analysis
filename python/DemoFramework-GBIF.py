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

data = my_species.load_data("./Etheostoma_blennioides2382397.pkl") # or just load existing data into Species object

data.columns # all the columns available per observation

data['country'].unique().tolist()

data.shape # there are 7226 observations, 138 parameters per observation

data['vernacularName'].unique().tolist() # self-explanatory

data['decimalLatitude'].tail(10)

import numpy as np
data_cleaned = data.dropna(subset = ['decimalLatitude', 'decimalLongitude']) # drop records where data not available

data_cleaned.shape # less occurrence records now: 5223

data_cleaned['basisOfRecord'].unique()

# this many records with no decimalLatitude and decimalLongitude
import numpy as np
data[data['decimalLatitude'].isnull() & data['decimalLongitude'].isnull()].size

data[data['decimalLatitude'].isnull() & 
     data['decimalLongitude'].isnull() & 
     data['locality'].isnull() & 
     data['verbatimLocality'].isnull()]

data_cleaned[['dateIdentified', 'day', 'month', 'year']].head()

data_selected = data_cleaned[data_cleaned['year']>2010][['decimalLatitude','decimalLongitude', 'rightsHolder', 'datasetName']]

data_selected[~data_selected.datasetName.isnull()].head(10)

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

list(set(csv_data.columns.tolist()) - set(data.columns.tolist()))



