import requests
from requests.auth import HTTPBasicAuth
import json

# Connect to any maprdb rest gataway and obtain a token
mapr_rest_auth = 'https://172.16.9.42:8243/auth/v2/token'
headers = {'content-type': 'application/json'}
bearerToken = None

try:
    bearerToken = requests.post(
            mapr_rest_auth, 
            headers=headers, verify=False,
            auth=HTTPBasicAuth('testuser', 'testuser')
        ).json()
except requests.exceptions.ConnectionError as e:
    pass

# Construct a header around your jwt token, same as previous notebook
headers = { 
'content-type': 'application/json', 
'Authorization': 'Bearer '+bearerToken['token'] 
} 

# Supress warnings about the self-signed certificate of maprdb data access gateway, so we dont OOM the notebook browser
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# For the demo, we can define a function that retrieves the results back to notebook 
def retrieveDataFromMaprdb(tablename):
    rest_get_trades_url = 'https://172.16.9.238:8243/api/v2/table/%2Fuser%2Ftestuser%2Feth%2F'+tablename+'?limit=1'

    try:
        table = requests.get(
            rest_get_trades_url, 
            headers=headers, verify=False
        )
        return table
    except requests.exceptions.ConnectionError as e:
        pass

retrieved_table = retrieveDataFromMaprdb('all_transactions_table')
print(retrieved_table.json())

# for a more sustainable way to query with conditions, we can create a function
# appending localparams this way allows us to get around encoding issues for special characters

def retrieveFilteredDataFromMaprdb(tablename, condition, projection):
    rest_get_trades_url = 'https://172.16.9.42:8243/api/v2/table/%2Fuser%2Ftestuser%2Feth%2F'+tablename
    localparams='condition='+condition
    localparams+='&fields='+projection

    
    try:
        table = requests.get(
            rest_get_trades_url, 
            headers=headers, verify=False,
            params=localparams
        )
        return table
    except requests.exceptions.ConnectionError as e:
        pass

# let's query for the the guys really overpaying - 200x the usual price of gas
filtered_table = retrieveFilteredDataFromMaprdb("all_transactions_table",
                                                '{"$gt":{"gasPrice":8000000000000}}',
                                                "")
#filtered_table.json()

from web3 import Web3, HTTPProvider, IPCProvider

# connect to your geth node to convert wei to eth
gethRPCUrl='http://172.16.9.41:8545'
web3 = Web3(HTTPProvider(gethRPCUrl))

# query filtering for same overpaid transactions, only bringing back selected fields
filtered_table_projection = retrieveFilteredDataFromMaprdb("all_transactions_table",
                                                '{"$gt":{"gasPrice":8000000000000}}',
                                                "gasPrice,gas,hash")

# Create new empty json to hold enriched transactions (in a local dataframe)
PriceSanitizedMeow=[]
filtered_table_projection=filtered_table_projection.json()
for originalTrasanction in filtered_table_projection['DocumentStream']:
    
    # Add a new column 'ActualEtherUsed'
    originalTrasanction['ActualEtherUsed'] = originalTrasanction['gas'] * web3.fromWei(originalTrasanction['gasPrice'],unit='ether')
    
    # Append enhanced Transaction to the PriceSanitizedMeow
    PriceSanitizedMeow.append(originalTrasanction)

# Optional - print the enriched json to see what the data looks like
PriceSanitizedMeow

# Pretty it up and sort it locally
import pandas as pd
pd.set_option('display.max_colwidth', -1)
prettydf = pd.DataFrame(PriceSanitizedMeow)
prettydf['hash'] = 'https://etherscan.io/tx/'+prettydf['hash']
prettydf.sort_values(by=prettydf.columns[0], ascending=False)

