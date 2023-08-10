import csv
import json
import os
import pprint

import requests

# get api token and set authorization
api_token = os.environ['GITHUB_API_TOKEN']
headers = {'Authorization': f'token {api_token}'}

# set url to a graphql endpoint
url = 'https://api.github.com/graphql'

# add a json query
query = """
{
  organization(login: "nteract") {
    name
    repositories(first: 80) {
      nodes {
        name
        url
        description
        id
        pushedAt
        createdAt
      }
    }
  }
}
"""

# submit the request
r = requests.post(url=url, json={'query': query}, headers=headers)

result = r.json()
#result

import pandas as pd

repos = pd.DataFrame(result['data']['organization']['repositories']['nodes'], columns=['name', 'url', 'description', 'pushedAt', 'createdAt'])
#repos

# By repo name
sorted_df = repos.sort_values(by=['name'])
sorted_df

# by creation date
sorted_df = repos.sort_values(by=['createdAt'])
sorted_df

# Top 10 repos by latest activity (pushedAt)
sorted_df = repos.sort_values(by=['pushedAt'], ascending=False)
sorted_df.head(10)

# output data to a csv
# df.to_csv('issue_report.csv')

