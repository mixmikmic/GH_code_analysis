import json
import tempfile

import pandas as pd
from cachetools import cached

from eptools.talks.fetch import fetch_talks_json
from eptools.people.fetch import genderize

output_csv = 'talks_spreadsheet.csv'

talks = fetch_talks_json(status='accepted', conf='ep2016')

def loop_talks(talks):
    for type, session in talks.items():
        for id, talk in session.items():
            yield type, talk
            

def get_keys(adict, key_names):
    return {k: adict.get(k, '') for k in key_names}


def set_first_columns(df, first_columns):
    """ Return a `df` with the `first_columns` as first columns and the
    rest of columns in any order after these."""
    rest_cols  = tuple(set(df.columns) - set(first_columns))
    col_names  = tuple(first_columns) + rest_cols
    return df.reindex_axis(col_names, axis=1)


@cached(cache={})
def cached_gender(first_name):
    return genderize(first_name)

# fields of interest
foi = ('id', 'title', 'speakers', 'emails', 'adm_type', 'url', 'type')

sheet_data = {talk['id']: get_keys(talk, foi) for _, talk in loop_talks(talks)}

first_name = lambda x: x['speakers'].split(' ')[0]

_ = [talk.__setitem__('gender', cached_gender(first_name(talk))['gender'])
     for _, talk in sheet_data.items()]

df = pd.DataFrame.from_records(sheet_data).T

df = set_first_columns(df, foi)

df = df.sort_values(['type', 'id'], axis=0)

df.to_csv(output_csv)

# have to reload again because the result from genderize is not perfect. It was manually corrected.
gdf = pd.read_csv('Session Plannings 2016 - Tabellenblatt2.csv')

genders = [k['gender'].values[0] for g, k in gdf.groupby('emails')]

print('Male speakers:',   genders.count('male' ))
print('Female speakers:', genders.count('female'))

