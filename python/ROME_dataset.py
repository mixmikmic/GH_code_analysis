get_ipython().magic('matplotlib inline')
from __future__ import division
import os

import matplotlib.pyplot as plt
import seaborn as sns

from bob_emploi.lib import read_data

data_folder = os.getenv('DATA_FOLDER')

fiche_dicts = read_data.load_fiches_from_xml(os.path.join(data_folder,  'rome/ficheMetierXml'))
rome = [read_data.fiche_extractor(f) for f in fiche_dicts]

n_skills = [len(x['skills']) for x in rome]
ax = sns.distplot(n_skills, kde=False)
_ = ax.set_title('number of skills per job_group')

n_activities = [len(x['activities']) for x in rome]
ax = sns.distplot(n_activities, kde=False)
_ = ax.set_title('number of activities per job_group')

n_titles = [len(x['titles']) for x in rome]
ax = sns.distplot(n_titles, kde=False)
_ = ax.set_title('number of job titles per job_group')

