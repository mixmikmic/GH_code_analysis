import json

import requests

CRIME_SOCRATA_VIEW_ID = 'ijzp-q8t2'

def get_data_portal_url(view_id):
    return 'http://data.cityofchicago.org/api/views/{view_id}'.format(
      view_id=view_id)

def get_dataset_columns(view_id):
    """
    Get dataset field names from the Socrata API

    Returns:
        A dictionary that acts as a lookup table from column ID to column name

    """
    url = get_data_portal_url(view_id)
    meta_response = requests.get(url)
    if not meta_response.ok:
        meta_response.raise_for_status()

    meta = meta_response.json()
    return {c['id']: c['name'] for c in meta['columns']}

columns = get_dataset_columns(CRIME_SOCRATA_VIEW_ID)

for column_id, name in columns.items():
    print("{}: {}".format(column_id, name))

date_column_id, date_column_name = next((i, n) for i, n in columns.items() if n.lower() == "date")
print("Date column ID: {}".format(date_column_id))

def slugify(s, replacement='_'):
    return s.replace(' ', replacement).lower()

def get_clean_column_lookup(column_lookup):
    return {str(i): slugify(n) for i, n in column_lookup.items()}

human_columns = get_clean_column_lookup(columns)
import pprint
pprint.pprint(human_columns)

def humanize_columns(row, column_lookup):
    humanized = {}
    for column_id, value in row.items():
        try:
            humanized[column_lookup[column_id]] = value
        except KeyError:
            humanized[column_id] = value
    
    return humanized

from datetime import date, timedelta

def build_query(since_date, date_column_id, view_id):
    """
    Get a Socrata API query for all records updated after the last update

    Args:
       since_date (datetine.date): date object. All crimes since this date will be retrieved.
       date_column_id (str): String containing the column ID for the dates we'll filter on
       view_id (str): Socrata view ID for this dataset

    Returns:
        Dictionary that can be serialized into a JSON sring used as the POST
        body to the Socrata API

    """

    query = {
        'originalViewId': view_id,
        'name': 'inline filter',
        'query' : {
            'filterCondition': {
                'type': 'operator',
                'value': 'AND',
                'children' : [{
                    'type' : 'operator',
                    'value' : 'GREATER_THAN',
                    'children': [{
                        'columnId' : date_column_id,
                        'type' : 'column',
                    }, {
                        'type' : 'literal',
                        'value' : since_date.strftime('%Y-%m-%d'),
                    }],
                }],
            },
        }
    }
    return query

# Months are different lenghts.  Let's just find the date 30 days ago
today = date.today()
date_30_days_ago = today - timedelta(days=30)
query = build_query(date_30_days_ago, date_column_id, CRIME_SOCRATA_VIEW_ID)

import pprint
print("The query looks like this: ")
pprint.pprint(query)

import json
import requests

def get_rows_url(start, count):
    url_tpl = "https://data.cityofchicago.org/api/views/INLINE/rows.json?method=getRows&start={start}&length={length}"
    return url_tpl.format(
      start=start,
      length=count
    )

def get_rows(query, start=0, count=1000):
    url = get_rows_url(start, count)
    headers = { 'content-type' : 'application/json' }
    response = requests.post(url, data=json.dumps(query), headers=headers, verify=False)
    return response.json()
    
def transform_row(row, transforms):
    transformed_row = row
    for transform in transforms:
        transformed_row = transform(transformed_row)
    
    return transformed_row
    
def get_all_rows(query, transforms=[]):
    continue_fetching = True
    page_size = 1000
    start = 0
    
    while continue_fetching:
        rows = get_rows(query, start, page_size)
        if len(rows) < page_size:
            continue_fetching = False
            
        start += page_size
        
        for row in rows:
            yield(transform_row(row, transforms))
        
crimes = list(get_all_rows(query, transforms=[lambda r: humanize_columns(r, human_columns)]))    

import pprint

print("There are {} crimes since {}".format(len(crimes), date_30_days_ago.strftime("%Y-%m-%d")))

print("The first one looks like: ")
pprint.pprint(crimes[0])



