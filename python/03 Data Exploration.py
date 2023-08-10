# In this notebook, we are going to look through the HIV-1 protease and reverse transcriptase sequence data. 
# The goal is to determine a strategy for downsampling sequences for phylogenetic tree construction

from Bio import SeqIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

proteases = [s for s in SeqIO.parse('sequences/HIV1-protease.fasta', 'fasta')]
len(proteases)

rts = [s for s in SeqIO.parse('sequences/HIV1-RT.fasta', 'fasta')]
len(rts)

def extract_metadata(sequences):
    """
    The metadata structure is as such:
    [Subtype].[Country].[Year].[Name].[Accession]
    """
    prot_metadata = []
    for s in sequences:
        metadata = s.id.split('.')
        data = dict()
        data['subtype'] = metadata[0]
        data['country'] = metadata[1]
        data['year'] = metadata[2]
        data['name'] = metadata[3]
        data['accession'] = metadata[4]

        prot_metadata.append(data)

    return pd.DataFrame(prot_metadata).replace('-', np.nan)

rt_metadf = extract_metadata(rts)
protease_metadf = extract_metadata(proteases)

rt_metadf.to_csv('csv/RT-all_metadata.csv')
rt_metadf

protease_metadf.to_csv('csv/Protease-all_metadata.csv')
protease_metadf

rt_metadf['year'].value_counts().plot(kind='bar')
rt_metadf['year'].value_counts().to_csv('csv/RT-num_per_year.csv')

fig = plt.figure(figsize=(15,3))
rt_metadf['country'].value_counts().plot(kind='bar')
rt_metadf['country'].value_counts().to_csv('csv/RT-num_per_country.csv')

protease_metadf['year'].value_counts().plot(kind='bar')
protease_metadf['year'].value_counts().to_csv('csv/Protease-num_per_year.csv')

fig = plt.figure(figsize=(15,3))
protease_metadf['country'].value_counts().plot(kind='bar')
protease_metadf['country'].value_counts().to_csv('csv/Protease-num_per_country.csv')

# Code for downsampling.
# Recall that the metadata structure is as such:
# 
#     [Subtype].[Country].[Year].[Name].[Accession]
# 
# We will use a dictionary to store the downsampled sequences.

import numpy as np
from collections import defaultdict
from itertools import product

years = np.arange(2003, 2008, 1)
countries = ['US', 'BR', 'JP', 'ZA', 'ES']

proteases_grouped = defaultdict()
for year, country in product(years, countries):
    proteases_grouped[(year, country)] = []

# Group the sequences first.
for s in proteases:
    country = s.id.split('.')[1]
    try:
        year = int(s.id.split('.')[2])
    except ValueError:
        year = 0
    if country in countries and year in years:
        proteases_grouped[(year, country)].append(s)

import random

random.seed(1) # for reproducibility

# Perform the downsampling
proteases_downsampled = defaultdict(list)
for k, v in proteases_grouped.items():
    proteases_downsampled[k] = random.sample(v, 10)
    
# Write the downsampled sequences to disk.
protease_sequences = []
for k, v in proteases_downsampled.items():
    protease_sequences.extend(v)
    
SeqIO.write(protease_sequences, 'sequences/proteases_downsampled.fasta', 'fasta')





