from collections import defaultdict
import datetime as datetime
import json
import random

import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')

DATA_DIR = "../data/"

df = pd.read_csv(DATA_DIR + 'uw_supplier_data1201616_edited.csv', encoding="latin-1")
df = df.rename(columns=dict((c, c.strip()) for c in df.columns))
print("* " + "\n* ".join(df.columns))
df.head()

def numberize(i):
    """Convert a string like '100,000' into a float, like 100000.0
    
    Since this data has a lot of variants for zero (missing data,
    null, strings like 'n/a'), we regard any failed float conversion
    as an indicator that the value is 0. This seems to be correct
    based on examining the data.
    """
    try:
        return float(str(i).replace(',', ''))
    except:
        return 0.0

# Rename wordy columns to more palatable strings
col_rename_dict = {
    'Supplier Name': 'supplier_name',
    "CALCULATED Total Monthly Potable Water Production Reporting Month Gallons": "total_potable_gal",
    "CALCULATED Total Monthly Potable Water Production 2013 Gallons": "total_potable_gal_2013",
    "CALCULATED Monthly CII Reporting Month": "cii_gal",
    'Reporting Month': 'reporting_month',
    'Total Population Served': 'total_population_served',
    '% Residential Use': 'percent_residential_use',
    'Hydrologic Region': 'hydrologic_region',
    'Conservation Standard (starting in June 2015) *Adjusted in March 2016 **Revised in June 2016': 'conservation_standard'
}
cols = list(df.columns)
for c in cols:
    if c not in col_rename_dict:
        del df[c]
df = df.rename(columns=col_rename_dict)

# Convert numerical columns from there comma-delimeted and other funky formats
numerical_columns = [c for c in df.columns if '_gal' in c] + ['percent_residential_use', 'total_population_served']
for c in numerical_columns:
    df[c] = df[c].apply(numberize)

# Compute a bunch of useful columns. Water usage breakdowns, etc.

df['reporting_month'] = pd.to_datetime(df['reporting_month'])
df['month'] = df['reporting_month'].apply(lambda x: x.month)
df['year'] = df['reporting_month'].apply(lambda x: x.year)
df['water_year'] = df['year'] + (df['month'] > 9).astype(int)

# Weirdly, the "total potable water REPORTED" number includes agricultural
# water, whereas the "total potable gallons CALCULATED" does not. The former
# is also reported in a range of units, while the latter converts to gallons.

df['residential_gal'] = df['total_potable_gal'] * (df['percent_residential_use'] / 100.0)
df['other_gal'] = df['total_potable_gal'] * (1 - df['percent_residential_use'] / 100.0) - df['cii_gal']

df['conservation_standard'] = df['conservation_standard'].apply(
    lambda s: 0.0 if pd.isnull(s) else float(s.strip('%')) / 100.0)

df.head()

def computeUsage(df):
    '''Given a dataframe for a single provider, create usage dict.'''
    usage = defaultdict(dict)
    for i, row in df.iterrows():
        m = row['reporting_month'].strftime('%Y-%m')
        pop = float(row['total_population_served'])
        usage['totalPerCapita'][m] = numberize(row['total_potable_gal']) / pop
        usage['residentialPerCapita'][m] = numberize(row['residential_gal']) / pop
        usage['commercialIndustrialPerCapita'][m] = numberize(row['cii_gal']) / pop
        usage['otherPotablePerCapita'][m] = numberize(row['other_gal']) / pop

    return dict(usage)  # Convert from defaultdict to regular dict

def computePredictions(usage):
    '''Compute monthly usage predictions for the upcoming 12 months.'''
    predictions = {}
    for month in range(1, 13):
        matching_keys = [k for k in usage if k.endswith("%02d" % month)]
        if len(matching_keys) == 0:
            # We have never seen data for this month, can't predict anything
            result = None
        elif len(matching_keys) == 1:
            # Only one occurance of this month, just predict homeostasis
            result = usage[matching_keys[0]]
        else:
            # Assume the year-over-year growth for this month is static.
            matching_keys.sort()
            growth = usage[matching_keys[-1]] / usage[matching_keys[-2]]
            result = usage[matching_keys[-1]] * growth
        predictions['%02d' %  month] = result
    return predictions

def computeTargetUsage(df, year):
    '''Given a dataframe for a single provider, calculate it's usage target for the year.'''
    if len(df[df['water_year'] == year]) < 12:
        print("Don't have complete data for water year %d, skipping..." % (year))
        return None
    reduction = df.loc[datetime.datetime(year - 1, 10, 15), 'conservation_standard']
    df_year = df[df['water_year'] == year]
    used_year = (df_year['total_potable_gal_2013'] / df_year['total_population_served']).sum()
    return (1 - reduction) * used_year

# One dataset identifies water suppliers by ID, the other merely by name. We need to pair them.

providers = pd.read_csv(DATA_DIR + 'provider_ids.tsv', sep='\t')
provider_id_lookup = {}
for p, i in zip(providers['Provider'], providers['ID']):
    provider_id_lookup[p.lower()] = i

js = {}
for i, items in enumerate(df.groupby("supplier_name").groups.items()):
    name, indices = items
    print(i, name)
    name = name.lower()
    if name not in provider_id_lookup:
        print("Can't find supplier ID for '%s', skipping..." % name)
        continue

    supplier_df = df.loc[indices].set_index('reporting_month', drop=False)
    usage = computeUsage(supplier_df)
    total = usage['totalPerCapita']
    del usage['totalPerCapita']
    preds = computePredictions(total)
    target_2016 = computeTargetUsage(supplier_df, 2016)
    target_2015 = computeTargetUsage(supplier_df, 2015)
    if target_2016 is None or target_2015 is None:
        continue

    js[provider_id_lookup[name]] = {
        "agencyName": name,
        "totalUsage": total,
        "usageByCategory": usage,
        "monthlyPrediction": preds,
        "annualTarget": target_2016,
        "previousTarget": target_2015,
    }

js[provider_id_lookup['East Bay Municipal Utilities District'.lower()]]

with open(DATA_DIR + 'usage.json', 'w') as f:
    f.write(json.dumps(js, indent=4))



