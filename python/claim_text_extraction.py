from __future__ import print_function
from google.cloud import bigquery
import pandas as pd
import matplotlib.pylab as plt
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import uuid
import time
# Variables to be used throughout the notebook, replace with your project
# and desired dataset name.
PROJECT_ID = 'my_project_name'
DEST_DATASET = 'claims_analysis'

# Create a python client we can use for executing table creation queries
client = bigquery.Client(project=PROJECT_ID)
# Create an HTTP client for additional functionality.
credentials = GoogleCredentials.get_application_default()
http_client = discovery.build('bigquery', 'v2', credentials=credentials)

get_ipython().magic('matplotlib inline')

dataset_ref = client.dataset(DEST_DATASET)
dataset = bigquery.Dataset(dataset_ref)
client.create_dataset(dataset)
print('Dataset {} created.'.format(dataset.dataset_id))

# Load a list of patents from disk
df = pd.read_csv('./data/20k_G_and_H_publication_numbers.csv')
df.head(5)

# Upload these to our dataset on BigQuery.
pubs_table = 'claim_text_publications'
full_table_path = '{}.{}'.format(DEST_DATASET, pubs_table)
df.to_gbq(destination_table=full_table_path,
          project_id=PROJECT_ID,
          if_exists='replace')

def create_table_as(query_string, dest_table, project_id=PROJECT_ID, dest_dataset=DEST_DATASET,
                    http_client=http_client, overwrite=True, use_legacy_sql=False):
  """Simulates a 'create table as' statement in BigQuery.

  Args:
    query_string: A string query that produces rows of data.
    dest_table: string table name to use for storing results.
    project_id: string project id to use for running query and storing results.
    dest_dataset: String name of dataset to use for storing results.
    http_client: An http client for use in inserting the BigQuery Job.
    overwrite: Should new data be appended to existing data or overwritten?
    use_legacy_sql: Defaults to standard_sql, but option to use legacy.

  Raises:
    Exception: If the BigQuery job finshes with an error, a general Exception is raised
    with the error message from BigQuery included.
  """
  write_disposition = 'WRITE_TRUNCATE'
  if not overwrite:
    write_disposition = 'WRITE_APPEND'
  config = {
      'kind': 'bigquery#job',
      'projectId': str(uuid.uuid4()),
      'configuration': {
          'query': {
              'query': query_string,
              'destinationTable': {
                  'projectId': project_id,
                  'datasetId': dest_dataset,
                  'tableId': dest_table
              },
              'createDisposition': 'CREATE_IF_NEEDED',
              'writeDisposition': write_disposition,
              'useLegacySql': use_legacy_sql
          }
      }
  }
  done = False
  request = http_client.jobs().insert(
      projectId=project_id, body=config).execute()
  job_id = request['jobReference']['jobId']
  while not done:
    jobs = http_client.jobs().list(projectId=project_id).execute()
    matches = [j for j in jobs['jobs'] if j['jobReference']['jobId'] == job_id]
    if matches[0]['state'] == 'DONE':
      if matches[0].get('errorResult'):
        raise Exception(
            'Create table failed: {}'.format(matches[0]['errorResult']))
      done = True
    print('Job still running...')
    time.sleep(5)

  print('Created table {}.'.format(dest_table))

query = """
#standardSQL

WITH P AS (
  SELECT 
  DISTINCT publication_number, 
  substr(cpc.code, 1,4) cpc4,
  floor(priority_date / 10000) priority_yr
  FROM `patents-public-data.patents.publications`,
  unnest(cpc) as cpc
  WHERE substr(cpc.code, 1,4) = 'G06F'
  AND floor(priority_date / 10000) >= 1995
  AND country_code = 'US'
)

SELECT 
P.publication_number,
P.priority_yr,
P.cpc4,
claims.text
FROM `patents-public-data.patents.publications` as pubs,
UNNEST(claims_localized) as claims
JOIN P 
  ON P.publication_number = pubs.publication_number
JOIN `{}.{}.{}` my_pubs
  ON pubs.publication_number = my_pubs.publication_number
WHERE claims.language = 'en'
""".format(PROJECT_ID, DEST_DATASET, pubs_table)
create_table_as(query, '20k_G06F_pubs_after_1994')

query = """
SELECT *
FROM `{}.{}.20k_G06F_pubs_after_1994`
WHERE RAND() < 500/20000 
""".format(PROJECT_ID, DEST_DATASET)
df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

# Quick visualization of the first 5 rows
df.head()

# Do we have a mix of data from 1995-present?
df.priority_yr.value_counts().sort_index()

# What do the claims look like?
for claim in df.sample(2).text:
  print('-----' * 5, '\n\nEXAMPLE CLAIM: \n\n', claim[:3000])

# The JS script to run against the text of our claims.
js = r"""

// Regex to find a period followed by any number of spaces and '2.' or '2 .'
var pattern = new RegExp(/[.][\\s]+[2][\\s]*[.]/, 'g');
if (pattern.test(text)) {
  return text.split(pattern)[0];
}

// If none of the above worked, try to find a reference to claim 1.
if (text.indexOf('claim 1 ') > -1) {
  return text.split('claim 1 ')[0];
}

// Look for claim cancelations and return first non-canceled claim
text = text.replace(/canceled/i, 'canceled')
text = text.replace(/cancelled/i, 'canceled')
if (text.indexOf('(canceled)') > -1) {
	text = text.split('(canceled)')
	canceled = text[0];
	not_canceled = text[1];
	// Determine first non-cancelled claim
	if (canceled.indexOf('-') > -1 ) {
		next_claim = parseInt(canceled.split('-')[1]) + 1
	}
	else {
		next_claim = 2;
	}
	// Split on next_claim + 1
	return not_canceled.split(next_claim + 1)[0].trim();
}


// If none of the above worked, try to find a sentence starting with 'The'
// This should only happen after claim 1 has been defined.
if (text.indexOf(' The ') > -1) {
  return text.split(' The ')[0];
}

// if none of the above worked return the first 2000 characters. 
return text.slice(0,2000);
"""

# Insert the JS code above into this string, and pass it to Big Query
query = r'''
#standardSQL
CREATE TEMPORARY FUNCTION get_first_claim(text STRING)
RETURNS STRING
LANGUAGE js AS """
{}
"""; 

 
SELECT 
pubs.*, 
get_first_claim(text) first_claim
FROM `{}.{}.20k_G06F_pubs_after_1994` pubs
'''.format(js, PROJECT_ID, DEST_DATASET)
CLAIM_TEXT_TABLE = '20k_G06F_pubs_after_1994_split_first_claim'
create_table_as(query, CLAIM_TEXT_TABLE)

query = """
SELECT * FROM `{}.{}.{}` WHERE RAND() < 500/20000 
""".format(PROJECT_ID, DEST_DATASET, CLAIM_TEXT_TABLE)
df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

for idx, row in df.sample(2).iterrows():
  print('------' * 10, '\nORIGINAL CLAIM TEXT:\n', '------' * 10)
  print(row.text[:2000], '\n')
  print('------' * 10, '\nRESULT AFTER PARSING WITH UDF:\n', '------' * 10)
  print(row.first_claim, '...\n\n')

# Average Word Count by Year:
query = r"""
#standardSQL
with words as (
  SELECT 
  publication_number,
  priority_yr,
  SPLIT(REGEXP_REPLACE(first_claim, r'\s{2,}', ''), ' ') word
  FROM `%s.%s.%s`
)

SELECT 
  priority_yr, 
  avg(num_words) avg_word_cnt
FROM (
  SELECT
  publication_number,
  priority_yr,
  count(*) as num_words
  from words, unnest(word)
  group by 1,2
)
GROUP BY 1
ORDER BY 1
""" %(PROJECT_ID, DEST_DATASET, CLAIM_TEXT_TABLE)
df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

plt.figure(figsize=(14, 6))
plt.plot(df.priority_yr, df.avg_word_cnt)
plt.xlabel('Priority Year', fontsize=14)
plt.xlim(1995, 2016)
plt.ylabel('Average Word Count')
plt.title('Average Word Count for the First Independent Claim of\nG06F Patents,'
          '1995 to present', fontsize=16)
plt.show()

query = """
#standardSQL
with elements as (
  SELECT 
  publication_number,
  priority_yr,
  SPLIT(first_claim, ';') element
  FROM `%s.%s.%s`
)

SELECT 
  priority_yr, 
  avg(num_elements) avg_element_cnt
FROM (
  SELECT
  publication_number,
  priority_yr,
  count(*) as num_elements
  from elements, unnest(element)
  group by 1,2
)
GROUP BY 1
ORDER BY 1
""" %(PROJECT_ID, DEST_DATASET, CLAIM_TEXT_TABLE)
df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard')

plt.figure(figsize=(14, 6))
plt.plot(df.priority_yr, df.avg_element_cnt, color='g')
plt.xlabel('Priority Year', fontsize=14)
plt.xlim(1995, 2016)
plt.ylabel('Average Element Count')
plt.title('Average Element Count for the First Independent Claim of\n'
          'G06F Patents, 1995 to present', fontsize=16)
plt.show()



