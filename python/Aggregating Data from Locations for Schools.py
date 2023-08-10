import pandas as pd
import numpy as np

tract_data = pd.read_csv('./CensusData/sfo data/census_data_plus_affordability.csv')

print tract_data.columns[-20:]

lats = tract_data.loc[:, ['INTPTLAT', 'INTPTLONG']]



new_columns = tract_data.columns.values
new_columns[np.flatnonzero(tract_data.columns.str.startswith('INTPTLONG'))[0]] = 'INTPTLONG'
tract_data.columns = new_columns
lats = tract_data.loc[:, ['INTPTLAT', 'INTPTLONG']]
tract_data.to_csv('./CensusData/sfo data/new_tract_data.csv')

from density_estimation import *

pts = lats.loc[20, :].values
pts

pub_schools = pd.read_csv('./Schools/schools_public_pt.csv')

df_schools_loc = pub_schools['the_geom'].str.extract('POINT \((.*) (.*)\)')

df_schools_loc = df_schools_loc.iloc[:, ::-1]

df_schools_loc.columns = ['Latitude', 'Longitude']

df_schools_loc = df_schools_loc.astype(float)

scores = np.zeros(lats.shape[0])
for i, p in enumerate(lats.values):
    scores[i] = point_density(p, (df_schools_loc['Latitude'], df_schools_loc['Longitude']), 
                              kernel='gaussian', bw=0.01)
scores *= 100. / scores.max()

from bqplot import pyplot as pl

pl.figure()
pl.hist(scores)
pl.show()

tract_data['pub_school_score'] = scores

private_schools = pd.read_csv('./Schools/pr_schools.csv')
private_schools.head()

pr_scores = np.zeros(lats.shape[0])
for i, p in enumerate(lats.values):
    pr_scores[i] = point_density(p, (private_schools['Latitude'], private_schools['Longitude']), 
                              kernel='gaussian', bw=0.01)
pr_scores *= 100. / pr_scores.max()

pl.figure()
pl.hist(pr_scores)
pl.show()

tract_data['pr_school_score'] = pr_scores

tract_data.to_csv('./CensusData/sfo data/tract_data_with_schools.csv', index=False)

pd.read_csv('./CensusData/sfo data/tract_data_with_schools.csv')



