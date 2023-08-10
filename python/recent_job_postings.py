get_ipython().magic('matplotlib inline')

import datetime
from os import path

import pandas as pd
import seaborn as _

from bob_emploi.lib import plot_helpers

DATA_PATH = '../../../data/'

pd.options.display.max_rows = 999

postings = pd.read_csv(
    path.join(DATA_PATH, 'recent_job_offers.csv'),
    dtype={'POSTCODE': str},
    parse_dates=['CREATION_DATE', 'MODIFICATION_DATE'], dayfirst=True, infer_datetime_format=True,
    low_memory=False)
postings.columns = postings.columns.str.lower()

postings.head(2).transpose()

_ = plot_helpers.hist_in_range(postings.creation_date, datetime.datetime(2014, 8, 1))

_ = postings.creation_date[postings.creation_date > datetime.datetime(2016, 8, 1)].value_counts().plot()

postings['activity_group'] = postings['activity_code'].str[:2]
def _percent_of_uniques(fields, postings):
    dupes = postings[postings.duplicated(fields, keep=False)]
    return 100 - len(dupes) / len(postings) * 100
'%.02f%%' % (_percent_of_uniques(['creation_date', 'rome_profession_code', 'postcode', 'activity_group'], postings))

'%.02f%%' % (_percent_of_uniques([
    'creation_date',
    'rome_profession_code',
    'postcode',
    'activity_group',
    'departement_code',
    'contract_type_code',
    'annual_minimum_salary',
    'annual_maximum_salary',
    'qualification_code',
    ],postings))

_PE_SEARCH_PAGE = (
    'https://candidat.pole-emploi.fr/candidat/rechercheoffres/resultats/'
    'A__COMMUNE_%(postcode)s_5___%(activity_group)s-_____%(rome_profession_code)s____INDIFFERENT_______________________')
for s in postings.sample().itertuples():
    print(_PE_SEARCH_PAGE % s._asdict())
    print(s.creation_date)

