get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb as PandasPdb
import csv

bioassembly = input("Please enter biological assembly number. Enter 0 if not reading biological assembly: ")
angstrom_limit = float(input("Please enter ångström limit: "))

#Loads in DataFrame of existing data from previous clustering script
df_frequency = pd.read_csv("BA" + bioassembly + "_" + str(angstrom_limit) + "Å_clustering_flavoproteins.csv", index_col=0)

#Change cluster_labels_kmeans number to desired cluster
df_frequency[df_frequency.cluster_labels_kmeans == 0]



