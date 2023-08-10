#Enter credentials

#enter Quovo username/password
quovo_username = ""
quovo_password = ""

#enter name and email of user
quovo_name = ""
quovo_email = ""

#enter Brokerage ID assoicated by quovo, 21534 - Test Data Brokerage
brokerage_ID = 21534

#enter brokerage username and password
brokerage_username = "testusername"
brokerage_password = "testpass"

#enter Investment Portfolio credentials
IP_W_username=""
IP_W_password=""
IP_R_username=""
IP_R_password=""

#Metrics
get_ipython().system('pip install -q metrics-tracker-client')
import metrics_tracker_client
metrics_tracker_client.DSX('IBM/Integrate-Investment-Portfolio')

#Get an Access Token

import requests
import json
import os

#make the request
print ("Get Quovo Access Token")
BASEURL = "https://api.quovo.com/v2/tokens"
headers = {
    'Content-Type': "application/json"
    }
data = {
    'name': "main_token"
    }
get_data = requests.post(BASEURL, auth=(quovo_username, quovo_password), headers=headers, data=json.dumps(data))
data = get_data.json()
json_data = json.dumps(data)

#if access token present, create 'token.json' file and assign token variable
if 'access_token' in data:
    f = open("token.json", "w")
    f.write(json_data)
    f.close()

    token = data["access_token"]["token"]
    print (json.dumps(data, indent=4, sort_keys=True))

#if access token present, create 'token.json' file and assign token variable
if 'access_token' in data:
    f = open("token.json", "w")
    f.write(json_data)
    f.close()

    token = data["access_token"]["token"]
    print (json.dumps(data, indent=4, sort_keys=True))

#else if token name in use, get token from 'token.json'
elif data["message"] == "The given name is already in use." and os.path.isfile('token.json'):
    print ("Get token from token.json")
    with open('token.json') as data_file:
        token_data = json.load(data_file)
        token = token_data["access_token"]["token"]
#else print status and message
else:
    print (json.dumps(data, indent=4, sort_keys=True))
    print ("status: " + str(data["status"]))
    print ("message: " + str(data["message"]))
    
print ("token: " + token)

#Create/Get user

#make the request to create user
BASEURL = "https://api.quovo.com/v2/users"
headers = {
    'Authorization': "Bearer " + token,
    'Content-Type': "application/json"
    }
data = {
    'username': "main_token",
    'name': quovo_name,
    'email': quovo_email
    }

get_data = requests.post(BASEURL, headers=headers, data=json.dumps(data))
data = get_data.json()
json_data = json.dumps(data)

#if user is created, retrieve the User ID
if 'user' in data:        
    print ("Create User")
    print (json.dumps(data, indent=4, sort_keys=True))    
    user_ID = data["user"]["id"]    
    print ("User ID: " + str(user_ID))

#else get all users, and assign User ID
else:    
    headers = {
        'Authorization': "Bearer " + token
        }
    get_user_data = requests.get(BASEURL, headers=headers)
    user_data = get_user_data.json()
    if 'users' in user_data:        
        print ("Get User")
        print (json.dumps(user_data, indent=4, sort_keys=True))
        user_ID = user_data["users"][0]["id"]
        print ("User ID: " + str(user_ID))
    else:
        print ("status: " + str(data["status"]))
        print ("message: " + str(data["message"]))

#Create/Get account

#make the request to create account
BASEURL = "https://api.quovo.com/v2/users/" + str(user_ID) + "/accounts"
headers = {
    'Authorization': "Bearer " + token,
    'Content-Type': "application/json"
    }
data = {
    'brokerage': brokerage_ID,
    'username': brokerage_username,
    'password': brokerage_password
    }
get_data = requests.post(BASEURL, headers=headers, data=json.dumps(data))
data = get_data.json()

#if account is created, retrieve the Account ID
if 'account' in data:
    print ("Create account")
    account_id = data["account"]["id"]
    print (json.dumps(data, indent=4, sort_keys=True))

#else if the account exists, then get accounts
elif data["id"] == "duplicate_account":
    print ("Get Account")
    BASEURL = "https://api.quovo.com/v2/accounts"
    headers = {
        'Authorization': "Bearer " + token
        }
    get_account_data = requests.get(BASEURL, headers=headers)
    account_data = get_account_data.json()

    print (json.dumps(account_data, indent=4, sort_keys=True))
    
    #find the account with the same brokerage and assign Account ID
    for accounts in account_data['accounts']:
        if accounts['brokerage'] ==  brokerage_ID:
            account_id = accounts["id"]

#else print the returned status and message
else:
    print ("status: " + str(data["status"]))
    print ("message: " + str(data["message"]))

print ("Account ID: " + str(account_id))

#Sync account

#make the request
print ("Sync account")
BASEURL = "https://api.quovo.com/v2/accounts/" + str(account_id) + "/sync"
headers = {
    'Authorization': "Bearer " + token
    }
get_data = requests.post(BASEURL, headers=headers)

#print json data
data = get_data.json()
print (json.dumps(data, indent=4, sort_keys=True))

# Check sync till status: good

#make the request
print ("Check Sync")
BASEURL = "https://api.quovo.com/v2/accounts/" + str(account_id) + "/sync"

headers = {
    'Authorization': "Bearer " + token
    }
get_data = requests.get(BASEURL, headers=headers)

#print json data
data = get_data.json()
print (json.dumps(data, indent=4, sort_keys=True))

# Get Portfolios

print ("Get Portfolios - (Pick first portfolio if multiple portfolios in the account)")

#make the request
BASEURL = "https://api.quovo.com/v2/accounts/" + str(account_id) + "/portfolios"
headers = {
    'Authorization': "Bearer " + token
    }
get_data = requests.get(BASEURL, headers=headers)

#print portfolios json data
portfolios_data = get_data.json()
print (json.dumps(portfolios_data, indent=4, sort_keys=True))

#retrieve Portfolio ID
portfolio_id = portfolios_data["portfolios"][0]["id"]
print ("Portfolio ID: " + str(portfolio_id))

#Get Portfolio Positions

#make the request
print ("Get Positions of Portfolio")
BASEURL = "https://api.quovo.com/v2/portfolios/" + str(portfolio_id) + "/positions"
headers = {
    'Authorization': "Bearer " + token
    }
get_data = requests.get(BASEURL, headers=headers)

#print positions json data
positions_data = get_data.json()
print (json.dumps(positions_data, indent=4, sort_keys=True))

#Load Investment Portfolio with brokerage portfolio data

import datetime

print ("Add portfolio to Investment Portfolio service")

#create timestamp
timestamp = '{:%Y-%m-%dT%H:%M:%S.%fZ}'.format(datetime.datetime.now())    

#assign portfolio name and brokerage name
IP_portfolio_name = portfolios_data["portfolios"][0]['portfolio_name']
IP_brokerage_name = portfolios_data["portfolios"][0]['brokerage_name']

print ("Investment Portfolio - Name: " + IP_portfolio_name)
print ("Investment Portfolio - Brokerage: " + IP_brokerage_name)

#make request for portfolio  
BASEURL = "https://investment-portfolio.mybluemix.net/api/v1/portfolios"
headers = {
        'Content-Type': "application/json",
        'Accept': "application/json"
        }

data = {
    'name': IP_portfolio_name,
    'timestamp': timestamp,
    'closed': False,
    'data': { 'brokerage': IP_brokerage_name }
    }
get_data = requests.post(BASEURL, auth=(IP_W_username, IP_W_password), headers=headers, data=json.dumps(data))

#print the status and returned json
status = get_data.status_code
print("Investment Portfolio status: " + str(status))

if status != 200:
    print(get_data)
else:
    data = get_data.json()
    print (json.dumps(data, indent=4, sort_keys=True))

#Load holdings into Investment Portfolio for a portfolio

print ("Load Investment Portfolio Holdings")

holdings_data = []

#read asset, quantity andd companyname from positions data and append the holdings array
for positions in positions_data['positions']:
    position_data = {}

    if 'ticker' in positions:
        position_data["asset"] = positions['ticker']
    if 'quantity' in positions:
        position_data["quantity"] = positions['quantity']
    if 'ticker_name' in positions:
        position_data["companyName"] = positions['ticker_name']

    if 'asset_class' in positions:
        if positions['asset_class'] != 'Cash':
            holdings_data.append(position_data)

#make the request
timestamp = '{:%Y-%m-%dT%H:%M:%S.%fZ}'.format(datetime.datetime.now())
BASEURL = "https://investment-portfolio.mybluemix.net/api/v1/portfolios/" + IP_portfolio_name + "/holdings"
headers = {
    'Content-Type': "application/json",
    'Accept': "application/json"
    }
data = {
    'timestamp': timestamp,
    'holdings': holdings_data,
    }
get_data = requests.post(BASEURL, auth=(IP_W_username, IP_W_password), headers=headers, data=json.dumps(data))

#print the status and returned json
status = get_data.status_code
print("Investment Portfolio Holding status: " + str(status))

if status != 200:
    print(get_data)
else:
    data = get_data.json()
    print (json.dumps(data, indent=4, sort_keys=True))

#View portfolio and holdings in Investment Portfolio

#make the request for portfolios
print ("Get Portfolios from Investment Portfolio")
BASEURL = "https://investment-portfolio.mybluemix.net/api/v1/portfolios/"
headers = {
    'accept': "application/json",
    'content-type': "application/json"
    }
get_data = requests.get(BASEURL, auth=(IP_R_username, IP_R_password), headers=headers)
print("Investment Portfolio status: " + str(get_data.status_code))

#print json data
data = get_data.json()
print (json.dumps(data, indent=4, sort_keys=True))


#make the request for holdings
print ("Get Portfolio Holdings for " + IP_portfolio_name)
BASEURL = "https://investment-portfolio.mybluemix.net/api/v1/portfolios/" + IP_portfolio_name + "/holdings?latest=true"
headers = {
    'accept': "application/json",
    'content-type': "application/json"
    }
get_data = requests.get(BASEURL, auth=(IP_R_username, IP_R_password), headers=headers)
print("Investment Portfolio - Get Portfolio Holdings status: " + str(get_data.status_code))

#print json data
data = get_data.json()
print (json.dumps(data, indent=4, sort_keys=True))



