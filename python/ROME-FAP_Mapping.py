from os import path
import codecs
import os
import pandas
from bob_emploi.lib import read_data

data_folder = os.getenv('DATA_FOLDER')
appellations = pandas.read_csv(path.join(data_folder, 'rome/csv/unix_referentiel_appellation_v330_utf8.csv'))
rome_names = pandas.read_csv(path.join(data_folder, 'rome/csv/unix_referentiel_code_rome_v330_utf8.csv'))
fap_names = read_data.parse_intitule_fap(path.join(data_folder, 'intitule_fap2009.txt'))
with codecs.open(path.join(data_folder, 'crosswalks/passage_fap2009_romev3.txt'), 'r', 'latin-1') as fap_file:
    fap_romeq_mapping = read_data.parse_fap_rome_crosswalk(fap_file.readlines())
# parse_fap_rome_crosswalk gives actually qualified ROME codes.
fap_romeq_mapping = fap_romeq_mapping.rename(columns={'rome': 'romeQ'})

fap_romeq_mapping['rome'] = fap_romeq_mapping['romeQ'].apply(lambda s: s[:5])
fap_mapping = fap_romeq_mapping.groupby(['rome','fap'], as_index=False).first()
del(fap_mapping['romeQ'])

flatten_mapping = fap_mapping.groupby('rome', as_index=False).agg({'fap': lambda x: sorted(x.tolist())})
flatten_mapping['fap1'] = flatten_mapping['fap'].apply(lambda x: x[0])
flatten_mapping['fap2'] = flatten_mapping['fap'].apply(lambda x: x[1] if len(x) > 1 else '')
flatten_mapping['fap3'] = flatten_mapping['fap'].apply(lambda x: x[2] if len(x) > 2 else '')
print('There is maximum %d FAP codes per ROME.' % flatten_mapping['fap'].str.len().max())
del(flatten_mapping['fap'])
flatten_mapping.head()

# Drop non ambiguous.
ambiguous_romes_fap_mapping = flatten_mapping[flatten_mapping['fap2'] != '']
ambiguous_romes_fap_mapping.head()

named_mapping = ambiguous_romes_fap_mapping
named_mapping = pandas.merge(
    named_mapping, fap_names,
    left_on='fap1', right_on='fap_code', how='left')
named_mapping.rename(columns={'fap_code': 'fap_code_1', 'fap_name': 'fap_name_1'}, inplace=True)
named_mapping = pandas.merge(
    named_mapping, fap_names,
    left_on='fap2', right_on='fap_code', how='left')
named_mapping.rename(columns={'fap_code': 'fap_code_2', 'fap_name': 'fap_name_2'}, inplace=True)
named_mapping = pandas.merge(
    named_mapping, fap_names,
    left_on='fap3', right_on='fap_code', how='left')
named_mapping.rename(columns={'fap_code': 'fap_code_3', 'fap_name': 'fap_name_3'}, inplace=True)

rome_name_clean = rome_names.rename(columns={'code_rome': 'rome', 'libelle_rome': 'rome_name', 'code_ogr': 'code_ogr_rome'})

named_mapping = pandas.merge(named_mapping, rome_name_clean, on='rome').fillna('')
named_mapping.head(2).transpose()

appellations_clean = appellations.rename(columns={'code_rome': 'rome'})

to_resolve = pandas.DataFrame(
    pandas.merge(appellations_clean, named_mapping, on='rome'),
    columns=[
        'rome', 'rome_name',
        'code_ogr', 'libelle_appellation_court',
        'fap_code_1', 'fap_name_1',
        'fap_code_2', 'fap_name_2',
        'fap_code_3', 'fap_name_3',
    ]).fillna('')

to_resolve.to_csv(path.join(data_folder, 'ambiguous_rome_fap.csv'))

to_resolve.head()

