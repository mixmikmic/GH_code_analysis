import requests
import urllib
import json
import pandas as pd

def get_table_data(table_ids):
    api_url = 'https://api.censusreporter.org/1.0/data/show/latest?'
    params = {'table_ids':','.join(table_ids),
             'geo_ids':'16000US3651000,860|16000US3651000',
              'primary_geo_id':'16000US3651000'}
    params_enc = urllib.urlencode(params)
    data = json.loads(requests.get(api_url + params_enc).text)
    return data

def get_table_as_json(table_ids):
    api_url = 'https://api.censusreporter.org/1.0/data/show/latest?'
    params = {'table_ids':','.join(table_ids),
             'geo_ids':'16000US3651000,860|16000US3651000',
              'primary_geo_id':'16000US3651000'}
    params_enc = urllib.urlencode(params)
    data = requests.get(api_url + params_enc).text
    return data

d = get_table_data(['B01001'])

d['data']['86000US07002']['B01001']['estimate']

# Adapted from https://gist.github.com/JoeGermuska/1ed425c068d540326854
def prep_for_pandas(json_data,include_moe=False):
    """Given a dict of dicts as they come from a Census Reporter API call, set it up to be amenable to pandas.DataFrame.from_dict"""
    result = {}
    for geoid, tables in json_data.iteritems():
        flat = {}
        for table, values in tables.iteritems():
            for kind, columns in values.iteritems():
                if kind == 'estimate':
                    flat.update(columns)
                elif kind == 'error' and include_moe:
                    renamed = dict((k+"_moe",v) for k,v in columns.iteritems())
                    flat.update(renamed)
        result[geoid] = flat
    return result

x = prep_for_pandas(d['data'])
x['86000US07002'] == d['data']['86000US07002']['B01001']['estimate']

pd.DataFrame(x)

def expand_column_names(col_dict):
    # Get the min and max indentation levels
    level_range = list(sorted(set(coldata['indent'] for colkey, coldata in col_dict.iteritems())))
    max_level, min_level = max(level_range), min(level_range)
    prev_level = min(level_range)
    curr_level = min(level_range)
    # loop through columns one at a time.
    # at each step, if we have increased the indent level,
    # add to the column prefix
    prefix = []
    out_names = {}
    for colkey in sorted(col_dict):
        coldata = col_dict[colkey]
        # print colkey, '=>', coldata['name']
        
        clean_name = coldata['name'].strip(':')
        
        if coldata['indent'] == min_level:
            prefix = [clean_name]
            out_names[colkey] =  ' '.join(prefix)
        elif coldata['indent'] > prev_level: #and coldata['indent'] != max_level:
            prefix.append(clean_name)
            out_names[colkey] = ' '.join(prefix)
        elif coldata['indent'] == prev_level:
            prefix.pop()
            prefix.append(clean_name)
            out_names[colkey] = ' '.join(prefix)
        elif coldata['indent'] < prev_level: # gone down a step
            prefix.pop() # remove the last item
            prefix.pop() # and the one before it
            prefix.append(clean_name)
            out_names[colkey] = ' '.join(prefix)
        else:
            out_names[colkey] = ' '.join(prefix + [clean_name])
        prev_level = coldata['indent']
    return out_names

# BUILD PANDAS DATAFRAME FROM CensusReporter TABLEID
def dataframe_from_json(table_name):
    d = get_table_data([table_name])
    df = pd.DataFrame.from_dict(prep_for_pandas(d['data']), orient='index')
    
    columns_in_order = list(sorted(df.columns))
    df = df[columns_in_order]

    columns_to_names = expand_column_names(d['tables'][table_name]['columns'])
    new_columns = [columns_to_names[colkey] for colkey in df.columns]

    df.columns = new_columns
    new_index = [rowname.split('US')[-1] for rowname in df.index]
    df.index = new_index
    df.index.name='ZIP/Loc Code'
    return df

# SAMPLE TABLES for NYC:
# B01001 - Demographics by Zip Code/Sex/Age
# B19013 - Household Income by Zip Code
# B25006 - Head of Household by Race/Ethnic Group
# EXAMPLE OF FUNCTION CALL => Insert specific Table ID (table_name) from CensusReporter

df = dataframe_from_json('B19013') # B01001 B25006
#df.head()

df.sort_values('Median household income in the past 12 months (in 2015 Inflation-adjusted dollars)', ascending=True)
#df.columns = zip(sorted(d['tables']['B01001']['columns'].keys()), expand_column_names(d['tables']['B01001']['columns']))

# Tests to make sure columns are lined up.
# May want to add some here.
#assert df['Total Female 15 to 17 years'].loc['3651000'] == d['data']['16000US3651000']['B01001']['estimate']['B01001030']

#assert df['Total Female'].loc['3651000'] == d['data']['16000US3651000']['B01001']['estimate']['B01001026']

columns_to_names = {col : d['tables']['B01001']['columns'][col]['name'] for col in d['tables']['B01001']['columns'].keys()}
for colkey in sorted(d['data']['86000US07036']['B01001']['estimate'].keys()):
    print columns_to_names[colkey], d['data']['86000US07036']['B01001']['estimate'][colkey]

d['data']['86000US07002']

# QUERY TABLE DATA & LOAD IT INTO A PANDAS DATAFRAME 
tbl_id = 'B01001'

df = pd.DataFrame(columns=create_column_multiindex(d['tables']['B01001']['columns']))

rows = []
for location_code in d['data']:
    if len(location_code.split('US')[-1]) == 5:
        zipcode = location_code.split('US')[-1]
        row = {'zip':zipcode}
        male_idx = list(sorted(d['data'][location_code][tbl_id]['estimate'].keys()))
        
        for colkey in sorted(d['data'][location_code][tbl_id]['estimate'].keys()):
            
            row[columns_to_names[colkey]] = d['data'][location_code][tbl_id]['estimate'][colname]
            #print columns_to_names[colname], "\t", d['data'][location_code][tbl_id]['estimate'][colname]
        rows.append(row)

# NYC Age & Male/Female Demographics by zip code         
df = pd.DataFrame(rows).groupby('zip').sum()
df.columns = [df.columns[-1]] + list(df.columns[:-1])
#df = df.transpose()
#df.tail()

df.T.loc['07036']

zip_codes = [x.split('US')[-1] for x in list((d['data']).iterkeys())]
print zip_codes

