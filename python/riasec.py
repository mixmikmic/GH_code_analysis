get_ipython().magic('matplotlib inline')

from __future__ import division
import glob
import json
import os
import itertools as it

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xmltodict
import numpy as np

from bob_emploi.lib import read_data

data_folder = os.getenv('DATA_FOLDER')

def riasec_dist(first, second):
    '''compute the distance between two characteristics on the hexagon'''
    if pd.isnull(first) or pd.isnull(second):
        return np.nan
    riasec = "RIASEC"
    a = riasec.find(first.upper())
    b = riasec.find(second.upper())
    assert a >= 0 and b >= 0
    return min( (a-b)%6, (b-a)%6)

# to call it on a dataframe row
riasec_dist_row = lambda row: riasec_dist(row.riasec_majeur, row.riasec_mineur)

fiche_dicts = read_data.load_fiches_from_xml(os.path.join(data_folder, 'rome/ficheMetierXml'))

fiches = pd.DataFrame(fiche['bloc_code_rome'] for fiche in fiche_dicts)
fiches['riasec_mineur'] = fiches.riasec_mineur.str.upper()
fiches['combined'] = fiches.riasec_majeur + fiches.riasec_mineur
fiches['riasec_dist'] = fiches.apply(riasec_dist_row, axis=1)

def visualize_codes(thing):
    '''Visualize the distribution of Holland codes
    major codes, minor codes, the combinations of both 
    and distances between
    '''
    riasec_counts = thing.riasec_majeur.value_counts().to_frame()
    riasec_counts['riasec_mineur'] = thing.riasec_mineur.value_counts()

    fig, ax = plt.subplots(3, figsize=(10, 10))
    riasec_counts.plot(kind='bar', ax=ax[0])
    thing.combined.value_counts().plot(kind='bar', ax=ax[1])
    thing.riasec_dist.hist(ax=ax[2])
    ax[0].set_title('Frequency of major and minor codes')
    ax[1].set_title('Frequency of major-minor combinations')
    ax[2].set_title('Histogram of hexagon distances')

    fig.tight_layout()
    
visualize_codes(fiches)

def extract(fiche):
    '''extract the base activities associated with a job fiche'''
    base_acts = fiche['bloc_activites_de_base']['activites_de_base']['item_ab'] 
    rome = {'rome_' + k: v for k, v in fiche['bloc_code_rome'].items()}
    return [dict(rome, **ba) for ba in base_acts]

fiche_acts = pd.DataFrame(sum(map(extract, fiche_dicts), []))
fiche_acts['riasec_mineur'] = fiche_acts.riasec_mineur.str.upper()
fiche_acts['rome_riasec_mineur'] = fiche_acts.riasec_mineur.str.upper()

combinations = it.product(['majeur', 'mineur'], ['majeur', 'mineur'])
for job, act in combinations:
    job_key = 'rome_riasec_' + job
    act_key = 'riasec_' + act
    match_count = (fiche_acts[job_key] == fiche_acts[act_key]).sum()
    fmt_str = "{} job fiche matches {} activity fiche in {:.2f}%"
    print(fmt_str.format(job, act, match_count / len(fiche_acts) * 100))

activities = pd.read_csv('../../../data/rome/csv/unix_referentiel_activite_v330_utf8.csv')
act_riasec = pd.read_csv('../../../data/rome/csv/unix_referentiel_activite_riasec_v330_utf8.csv')
acts = pd.merge(activities, act_riasec, on='code_ogr')

acts['riasec_mineur'] = acts.riasec_mineur.str.upper()
acts['combined'] = acts.riasec_majeur + fiches.riasec_mineur
acts['riasec_dist'] = acts.apply(riasec_dist_row, axis=1)

base_acts = acts[acts.libelle_type_activite == 'ACTIVITE DE BASE']

visualize_codes(acts)
visualize_codes(fiches) #for comparison

