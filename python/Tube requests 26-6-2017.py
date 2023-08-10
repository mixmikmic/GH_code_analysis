import requests
import json
import pprint

url = 'https://api.tfl.gov.uk/line/mode/tube/status'
r = requests.get(url)  # get the response object

response = r
filename = 'tube_state.json'

with open(filename, 'wb') as fd:                 # write the response to file. Can format the .json file but no blank lines allowed 
    for chunk in response.iter_content(chunk_size=128):
        fd.write(chunk)

def print_json(json_data):
    pprint.PrettyPrinter().pprint(json_data)

bakerloo_dict = r.json()[0]
print_json(bakerloo_dict)

bakerloo_dict.keys()

bakerloo_dict['lineStatuses'][0].keys()

bakerloo_dict['lineStatuses'][0]['statusSeverityDescription']

lines = [line['id'] for line in r.json()]
lines

statuses = [line['lineStatuses'][0]['statusSeverityDescription'] for line in r.json()]
statuses

line_statuses =  {key:value for key, value in zip(lines, statuses)}

line_statuses.keys()

line_statuses['district']

# convenience function to get dict of statuses
def tube_statuses():
    url = 'https://api.tfl.gov.uk/line/mode/tube/status'
    r = requests.get(url)  # get the response object
    lines = [line['id'] for line in r.json()]
    statuses = [line['lineStatuses'][0]['statusSeverityDescription'] for line in r.json()]
    return {key:value for key, value in zip(lines, statuses)}


tube_statuses()['piccadilly']



