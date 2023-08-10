get_ipython().magic('matplotlib inline')

import pandas
import warnings
from pandas.io.pytables import PerformanceWarning

warnings.simplefilter(action='ignore', category=PerformanceWarning)

RATES_URL = 'https://api.data.gov/gsa/calc/rates/'

FIELD_DEFAULTS = {
    'Search terms': '',
    'Minimum experience': 0,
    'Maximum experience': 45,
    'Education level': '',
    'Worksite': '',
    'Business size': '',
    'Schedule': '',
    'Contract year': 'current',
    'Proposed price': '',
    'Query type': 'match_all',
    'Sort': 'current_price',
    'Rows excluded': 0,
}

rows = pandas.read_csv('logs.csv', nrows=None, index_col=False, usecols=[
    'Time',
    'Method',
    'URL',
    'State',
    'Country',
    'City',
    'Status',
    'IP Address',
])

rows.drop_duplicates(subset=['Method', 'URL', 'IP Address'], inplace=True)

from urllib.parse import urlparse, parse_qsl

rates = rows
rates = rates[rates['Method'] == 'GET']
rates = rates[rates['Status'] == 200]
rates = rates[rates['URL'].str.startswith(RATES_URL)]
rates['Time'] = rates['Time'].apply(pandas.to_datetime)

del rates['Method']
del rates['Status']

queries = rates['URL'].apply(
    lambda url: dict(parse_qsl(urlparse(url).query))
).apply(pandas.Series)

rates['Search terms'] = queries['q'].str.lower().str.strip().str.replace(r',[ ]*$', '')
rates['Minimum experience'] = queries['min_experience'].apply(pandas.to_numeric)
rates['Maximum experience'] = queries['max_experience'].apply(pandas.to_numeric)
rates['Education level'] = queries['education']
rates['Worksite'] = queries['site']
rates['Business size'] = queries['business_size']
rates['Schedule'] = queries['schedule']
rates['Contract year'] = queries['contract-year']
rates['Proposed price'] = queries['proposed-price']
rates['Query type'] = queries['query_type']
rates['Sort'] = queries['sort']
rates['Rows excluded'] = queries['exclude'].str.split(',').str.len()

rates = rates.fillna(FIELD_DEFAULTS)

store = pandas.HDFStore('logs.h5')
store['rates'] = rates
store.close()

store = pandas.HDFStore('logs.h5')
rates = store['rates']
store.close()

rates['Search terms'].value_counts().head(30).to_frame('Number of searches')

total_rates = rates.shape[0]
percentages = {}

for field, default in FIELD_DEFAULTS.items():
    non_default_rates = rates[rates[field] != default].shape[0]
    percentages[field] = int(non_default_rates / total_rates * 100)

pandas.Series(percentages).to_frame('%').sort_values(by='%', ascending=False)  .plot(kind='barh', title="Search customization")

customized = pandas.Series(False, rates.index)
customization_factors = pandas.Series(0, rates.index)

for field, default in FIELD_DEFAULTS.items():
    if field == 'Search terms':
        continue
    field_was_not_default = rates[field] != default
    customized = customized | field_was_not_default
    customization_factors += field_was_not_default.astype(int)

customized_rates = rates[customized]

customized_rates.shape[0] / rates.shape[0]

customization_factors.hist(bins=len(FIELD_DEFAULTS))

get_ipython().magic('matplotlib inline')

rates[rates['Minimum experience'] > 0]['Minimum experience'].value_counts().sort_index().plot()

timerates = rates.set_index(['Time'])

r = pandas.DataFrame()

for term in ['engineer', 'analyst', 'manager', 'subject matter expert']:
    r[term] = (timerates['Search terms'].str.contains(term)).astype(int).resample('W').sum()

r.plot(title='Labor category queries per day')

(customized_rates['State'])  .value_counts()[:10].plot(kind='barh', title="Customized searches by state")



