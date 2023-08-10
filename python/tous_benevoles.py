import os
from os import path

import pandas as pd
import seaborn as _
import xmltodict

DATA_FOLDER = os.getenv('DATA_FOLDER')

dataset = xmltodict.parse(open(path.join(DATA_FOLDER, 'tous_benevoles.xml'), 'rb'))

type(dataset)

dataset.keys()

type(dataset['jobs'])

dataset['jobs'].keys()

type(dataset['jobs']['job'])

len(dataset['jobs']['job'])

jobs = pd.DataFrame(dataset['jobs']['job'])
jobs.head()

jobs.describe()

jobs[jobs.duplicated(subset='JobId', keep=False)].groupby('JobId').size().head()

all_post_codes = jobs.groupby('JobId').PostalCode.apply(lambda codes: ','.join(codes.sort_values()))
all_post_codes[all_post_codes.str.contains(',')].iloc[0]

jobs['isAvailableEverywhere'] = jobs.JobId.map(all_post_codes) == '06000,13001,31000,33000,34000,35000,44000,59000,67000,69001,75001'
jobs.sort_values('isAvailableEverywhere', ascending=False).head()

jobs[jobs.JobId == '35421'][list(set(jobs.columns) - {'isAvailableEverywhere'})].describe()

all_fields_but_geo = set(jobs.columns) - set(('City', 'PostalCode'))
jobs.drop_duplicates(subset=all_fields_but_geo)    .groupby('JobId')    .size()    .value_counts()

_ = jobs.groupby('City').size().sort_values(ascending=False).head(20).sort_values().plot(kind='barh')

jobs[jobs.City.str.startswith('PARIS')].City.value_counts()

jobs[(jobs.City.str.contains('[A-Z] [0-9]')) & ~(jobs.City.str.startswith('PARIS'))].City.value_counts()

jobs['clean_city'] = jobs['City'].str.replace(' \d+', '')
jobs.clean_city.value_counts().head()

_ = jobs[jobs.City != 'LEFFRINCKOUCKE'].    groupby('clean_city').size().sort_values(ascending=False).    head(20).sort_values().plot(kind='barh')

_ = jobs[jobs.City != 'LEFFRINCKOUCKE']    .groupby('clean_city').size().sort_values(ascending=False)    .reset_index(drop=True)    .plot(ylim=(0, 50))  # 50 is taken from the chart above.

jobs['departement'] = jobs.PostalCode.str[:2]
jobs_per_departement = jobs[jobs.City != 'LEFFRINCKOUCKE'].groupby('departement').size().sort_values(ascending=False)
_ = jobs_per_departement.head(20).plot(kind='bar')

_ = jobs_per_departement    .reset_index(drop=True)    .plot(ylim=(0, 100))  # 100 was taken from the chart above.

sum(jobs_per_departement >= 3)

jobs[['JobDescription', 'JobTitle']].drop_duplicates().head().transpose()

jobs.JobDescription.str.startswith('Mission proposée par ').value_counts()

jobs.JobTitle.str.startswith('Bénévolat : ').value_counts()

jobs['title'] = jobs.JobTitle.str.replace('^Bénévolat : ', '')
jobs['proposed_by'] = jobs.JobDescription.str.extract('^Mission proposée par ([^<]+)<br />', expand=False)
jobs['description'] = jobs.JobDescription.str.replace('^Mission proposée par ([^<]+)<br />', '')
jobs[['title', 'proposed_by', 'description']].drop_duplicates().head()

jobs.description.str.startswith('<b>Informations complémentaires</b>').value_counts()

jobs['description'] = jobs.description.str.replace('^<b>Informations complémentaires</b>', '').str.strip()
jobs[['title', 'proposed_by', 'description']].drop_duplicates().head()

