import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

FHRS_URL = 'https://opendata.camden.gov.uk/api/views/ggah-dkrr/rows.csv?accessType=DOWNLOAD'

fhrs = pd.read_csv(FHRS_URL)

fhrs['Rating Date'] = pd.to_datetime(fhrs['Rating Date'])

fhrs = fhrs[(fhrs['Business Type Description'] == 'Restaurant/Cafe/Canteen') &             (fhrs['Rating Value'] != 'Exempt') &             (fhrs['Rating Value'] != 'AwaitingInspection') &             (~fhrs['New Rating Pending'])]

fhrs['Rating Value'] = fhrs['Rating Value'].astype('int')

fhrs['Rating Value'].value_counts().sort_index().plot.bar()

sns.countplot(x='Rating Value', data=fhrs)

fhrs['Rating Year'] = fhrs['Rating Date'].dt.year

fhrs.boxplot(column='Rating Value', by='Rating Year')

sns.boxplot(x='Rating Year', y='Rating Value', data=fhrs)

scores = ['Hygiene Score', 'Structural Score', 'Confidence In Management Score', 'Rating Value']
pd.plotting.scatter_matrix(fhrs[scores])

sns.pairplot(fhrs[scores].dropna())

sns.regplot(x='Hygiene Score', y='Rating Value', data=fhrs, x_jitter=1.25, y_jitter=0.25)

