# just for presentation in notebooks
from pprint import pprint as print

# our actual code imports
from bravado.client import SwaggerClient


# The spec that will be used to generate the methods of the API client.
OPENTRIALS_API_SPEC = 'http://api.opentrials.net/v1/swagger.yaml'

# we want our data returned as an array of dicts, and not as class instances.
config = {'use_models': False}

# instantiate our API client
client = SwaggerClient.from_url(OPENTRIALS_API_SPEC, config=config)

# inspect the client properties
dir(client)

# Passing in a very simple query, we will paginate results by 10
# The query response is then saved in the `result` variable
result = client.trials.searchTrials(q='depression', per_page=10).result()

'OpenTrials knows about {} trials related to depression.'.format(result['total_count'])

[obj['public_title'] for obj in result['items']]

sample = result['items'][0]
list(sample.keys())

# the unique identifer in the OpenTrials database
sample['id']

# the globally known identifiers
sample['identifiers']

# Does the trial have published results?
sample['has_published_results'] or 'No results'

# when was the trial first registered?
sample['registration_date'].isoformat()

# exactly which conditions does the trial test
[condition['name'] for condition in sample['conditions']]

# and, which interventions are tested against this condition?
[intervention['name'] for intervention in sample['interventions']]

# is there any summary description that tells us what this trial is about?
sample['brief_summary']

# what data sources have contributed to the information on this trial?
list(sample['sources'].keys())

# what records from these sources has OpenTrials collected?
list([(record['source_url'], record['url'])for record in sample['records']])

dir(client)

sample = result['items'][9]

sample

sample['interventions']

intervention = client.interventions.getIntervention(id=sample['interventions'][0]['id']).result()

intervention

# the total number of trials
result = client.trials.searchTrials(per_page=10).result()
trial_count = result['total_count']

trial_count

# the total number of trials we think have discrepancies
result = client.trials.searchTrials(q='_exists_:discrepancies', per_page=10).result()
discrepancy_count = result['total_count']

discrepancy_count

sample = result['items'][0]

sample['public_title'], sample['status'], sample['has_published_results'], [i['name'] for i in sample['interventions']], sample['discrepancies'], sample['registration_date'].isoformat()

# this and that
result = client.trials.searchTrials(q='suicide AND depression', per_page=10).result()

result['total_count']

# wildcard matches
result = client.trials.searchTrials(q='head*', per_page=10).result()

result['total_count']

# this or that
result = client.trials.searchTrials(q='headache OR migraine', per_page=10).result()

result['total_count']

# fuzzy matching
result = client.trials.searchTrials(q='brain~', per_page=10).result()

result['total_count']

# date ranges
result = client.trials.searchTrials(q='registration_date:[2014-01-01 TO 2014-12-31]', per_page=10).result()

result['total_count']

# grouping clauses
result = client.trials.searchTrials(q='(male OR female) AND sex', per_page=10).result()

result['total_count']

