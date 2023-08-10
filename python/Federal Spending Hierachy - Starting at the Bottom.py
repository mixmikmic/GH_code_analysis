# For those interested in the API, here's an example of displaying that list of
# all awards with the "State and Local Air Quality" program activity classification

import json
import requests
import pandas as pd
from pandas.io.json import json_normalize

uri = 'https://api.usaspending.gov/api/v1/accounts/awards/?program_activity__program_activity_name=Categorical%20Grant:%20%20State%20and%20Local%20Air%20Quality%20Management&limit=500'
r = requests.get(uri)
air_awards  = pd.DataFrame(json_normalize(r.json()['results']))


# This is pretty good, but we can can show better info by using the award identifiers
# in the air quality award list to grab some additional details about each item
award_list = air_awards['award'].tolist()
uri = 'https://api.usaspending.gov/api/v1/awards/'
headers = {'content-type': 'application/json'}
payload = {
    "limit": 500, 
    "fields": [
        "id", "total_obligation", "description", "type", "type_description",
       "fain", "piid", "uri", "recipient", "place_of_performance"],
    "filters": [{"field": "id", "operation": "in", "value": award_list}]
}
r = requests.post(uri, data=json.dumps(payload), headers=headers)
air_award_details = pd.DataFrame(json_normalize(r.json()['results']))

air_merged = pd.merge(air_awards, air_award_details, left_on='award', right_on='id')
# as an example, just show the first 10 awards
air_merged[[
    'type_description', 'recipient.recipient_name', 'recipient.location.city_name',
    'program_activity.program_activity_code', 'program_activity.program_activity_name',
    'treasury_account.federal_account.account_title'
    ]].head(10)

