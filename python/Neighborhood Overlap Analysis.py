from __future__ import division, print_function

import numpy as np
import pandas as pd
import geopandas as gpd

dna_df = gpd.read_file("dna_neighborhoods.geojson")

dna_df.head()

# Remove invalid geometries, warnings refer to the geometries being removed
dna_df = dna_df[dna_df['geometry'].is_valid]
dna_df["area"] = dna_df["geometry"].area
dna_df['col_index'] = dna_df.index

# Create a csv with the counts of each neighborhood
merge_count = pd.DataFrame(pd.value_counts(dna_df['neighborhood']))
merge_count.reset_index(inplace=True)
merge_count.columns = ['neighborhood', 'count']
merge_count.to_csv('neighborhood_count.csv', index=False)
merge_count.head()

# Create copy of data frame to do cross join, clean up columns
dna_match = dna_df
dna_match = dna_df[['neighborhood', 'geometry', 'col_index']]
dna_match = dna_match.rename(columns={'geometry': 'geometry_y', 'col_index': 'col_index_y'})

# Create cross join on neighborhoods and drop any shapes matched against themselves
dna_merge = pd.merge(dna_df, dna_match, on='neighborhood')
dna_merge = dna_merge[dna_merge['col_index'] != dna_merge['col_index_y']]

# Define function to be applied against each row, more readable than lambda
def get_overlap(row):
    return row['geometry'].intersection(row['geometry_y']).area / row['area']

# Apply overlap function, group and write to csv
dna_merge['overlap'] = dna_merge.apply(lambda row: get_overlap(row), axis=1)

# Get the mean overlap for each neighborhood, and then bring back counts of shapes for each
dna_group = dna_merge.groupby(['neighborhood'], as_index=False)['overlap'].mean()
count_dna = pd.read_csv('neighborhood_count.csv')
dna_group = pd.merge(dna_group, count_dna, on='neighborhood')

dna_group.to_csv('neighborhood_overlap.csv', index=False)
dna_group.head()



