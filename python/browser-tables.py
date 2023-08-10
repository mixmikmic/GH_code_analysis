import os

import pandas
import sklearn.metrics
import json
import requests

base_url = 'https://github.com/dhimmel/learn/raw/d2251a942813015d0362a90f179c961016336e77/'
compound_df = pandas.read_table(base_url + 'summary/compounds.tsv')
disease_df = pandas.read_table(base_url + 'summary/diseases.tsv')
prob_df = pandas.read_table(base_url + 'prediction/predictions/probabilities.tsv')
metapath_df = pandas.read_table(base_url + 'prediction/features/metapaths.tsv')

url = 'https://github.com/dhimmel/disease-ontology/raw/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/term-names.tsv'
doid_synonym_df = pandas.read_table(url)
disease_to_names = {disease_id: ' | '.join(sorted(set(df.name))) for disease_id, df in doid_synonym_df.groupby('doid')}
disease_df['synonyms'] = disease_df.disease_id.map(disease_to_names)
disease_df.head(2)

url = 'https://github.com/dhimmel/drugbank/raw/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/aliases.json'
compound_to_aliases = requests.get(url).json()
compound_to_aliases = {k: ' | '.join(v) for k, v in compound_to_aliases.items()}
compound_df['synonyms'] = compound_df.compound_id.map(compound_to_aliases)
compound_df.head(2)

url = 'https://github.com/dhimmel/clintrials/raw/4d63098c79042b7048f546720e727bc94e232182/data/DrugBank-DO-slim-counts.tsv'
clintrial_df = pandas.read_table(url)
clintrial_df = clintrial_df[['compound_id', 'disease_id', 'n_trials']]
prob_df = prob_df.drop('n_trials', axis='columns').merge(clintrial_df, how='left')
prob_df.n_trials = prob_df.n_trials.fillna(0).astype(int)

prob_df.head(2)

def get_auroc(df):
    try:
        auroc = sklearn.metrics.roc_auc_score(y_true=df.status, y_score=df.prediction)
    except ValueError:
        auroc = None
    series = pandas.Series()
    series['auroc'] = auroc
    return series
    
compound_df = compound_df.merge(
    prob_df.groupby('compound_id').apply(get_auroc).reset_index()
)
disease_df = disease_df.merge(
    prob_df.groupby('disease_id').apply(get_auroc).reset_index()
)

# Add descriptions for compounds
url = 'https://github.com/dhimmel/drugbank/raw/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank-slim.tsv'
compound_desc_df = pandas.read_table(url).rename(columns={'drugbank_id': 'compound_id'})[['compound_id', 'description']]
compound_df = compound_df.merge(compound_desc_df)
compound_df.head(2)

# Add descriptions for diseases
url = 'https://github.com/dhimmel/disease-ontology/raw/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/description.tsv'
disease_desc_df = pandas.read_table(url)[['disease_id', 'description']]
disease_df = disease_df.merge(disease_desc_df, how='left')
disease_df.head(2)

def df_to_json(df, path, double_precision=6, indent=None):
    """Write a pandas dataframe to a JSON text file formatted as datatables input."""
    dump_str = df.to_json(orient='split', double_precision=double_precision, force_ascii=False)
    obj = json.loads(dump_str)
    del obj['index']
    with open(path, 'wt') as fp:
        json.dump(obj, fp, sort_keys=True, indent=indent)

path = os.path.join('browser-tables', 'compounds.json')
df_to_json(compound_df, path, indent=1)

path = os.path.join('browser-tables', 'diseases.json')
df_to_json(disease_df, path, indent=1)

path = os.path.join('browser-tables', 'metapaths.json')
df_to_json(metapath_df, path, indent=1)

prob_df = prob_df[[
    'compound_name',
    'disease_name',
    'prediction',
    'compound_percentile',
    'disease_percentile',
    'category',
    'n_trials',
    'compound_id',
    'disease_id',
]]

for compound_id, df in prob_df.groupby('compound_id'):
    path = os.path.join('browser-tables', 'compound', '{}.json'.format(compound_id))
    df = df.drop(['compound_id', 'compound_name'], axis = 'columns')
    df['synonyms'] = df.disease_id.map(disease_to_names)
    df = df.merge(disease_desc_df, how='left')
    df_to_json(df, path, indent=0)

for disease_id, df in prob_df.groupby('disease_id'):
    disease_id = disease_id.replace(':', '_')
    path = os.path.join('browser-tables', 'disease', '{}.json'.format(disease_id))
    df = df.drop(['disease_id', 'disease_name'], axis = 'columns')
    df['synonyms'] = df.compound_id.map(compound_to_aliases)
    df = df.merge(compound_desc_df, how='left')
    df_to_json(df, path, indent=0)

df.head(2)

info = {}

for kind, df in ('compound', compound_df), ('disease', disease_df):
    df = df.where(df.notnull(), None)
    for row in df.itertuples():
        row_id = getattr(row, kind + '_id')
        row_id = row_id.replace(':', '_')
        elem = [getattr(row, kind + '_name'), kind]
        item = {
            'name': getattr(row, kind + '_name'),
            'type': kind,
            'treats': int(row.treats),
            'palliates': int(row.palliates),
            'edges': int(row.total_edges),
            'description': row.description,
        }
        if pandas.notnull(row.auroc):
            item['auroc'] = round(row.auroc, 4)
        info[row_id] = item

with open('./browser-tables/info.json', 'w') as fp:
    json.dump(info, fp, indent=1, sort_keys=True)

