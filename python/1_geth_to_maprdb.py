import requests
from requests.auth import HTTPBasicAuth
import json

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

# Optional: print the bearer token to see what it looks like
bearerToken

headers = { 
'content-type': 'application/json', 
'Authorization': 'Bearer '+bearerToken['token'] 
} 
headers

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

transaction_put_url = 'https://172.16.9.42:8243/api/v2/table/%2Fuser%2Ftestuser%2Feth%2Fall_transactions_table'
response = None

try:
    response = requests.put(
            transaction_put_url, 
            headers=headers, verify=False
        )
    print(response)
except requests.exceptions.ConnectionError as e:
    pass

# Note: a 409 response means the table already exists (which is good if you're running this for the second time)
# 201 means table created successfully, and 401 is most likely caused by an expired token

# The following code connects to my geth container (replace with your own private IP).  
from web3 import Web3, HTTPProvider, IPCProvider

gethRPCUrl='http://172.16.9.41:8545'
web3 = Web3(HTTPProvider(gethRPCUrl))

# Optional - print out one block to see what the data looks like (or look at sample block_5399622.json file)
dict(web3.eth.getBlock(5399622))

# Define a function to retrieve all transactions for a given block
def getAllTransactions(block):
    allTransactions = []
    
    for transaction in dict(web3.eth.getBlock(block,full_transactions=True))['transactions']:
        allTransactions.append((dict(transaction)))
        
    return allTransactions

# Optional: print transactions for a block to see what the nested data looks like (and to make sure the function works)
getAllTransactions(5412388)

def getTransactionsAndInsertToDB(blockstart,blockend,txstable):
    for block in range(blockstart,blockend):
        txsLastBlock=getAllTransactions(block)
        
        #print("Inserting to maprdb")
        rest_put_txs_url = 'https://172.16.9.238:8243/api/v2/table/%2Fuser%2Ftestuser%2Feth%2F'+txstable
      
        try:
            for transaction in txsLastBlock:
                transaction['_id']=transaction['hash']
                #print(transaction)
                response = requests.post(
                    rest_put_txs_url, 
                    headers=headers, verify=False,
                    data=json.dumps(transaction)
            )
        except Exception as e:
            print(e)
            pass

# retrieve the latest block number, so we can get a recent range of blocks
currentblock = web3.eth.getBlock('latest').number

getTransactionsAndInsertToDB(blockstart=currentblock-100,
                           blockend=currentblock,
                           txstable="all_transactions_table")

