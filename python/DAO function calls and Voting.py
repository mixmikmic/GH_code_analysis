from etherscan.accounts import Account
from api_key import key
import pandas as pd
from datetime import datetime, timedelta
import json

api = Account()

transactions = api.get_all_transactions()

# Write to file to keep it up to date
with open('dao_transactions.json', mode='w') as f:
    f.write(transactions)

trans = pd.read_json('dao_transactions.json')

# trans = pd.DataFrame(transactions)
trans['value'] = trans.value.astype(float)
trans['blockNumber'] = trans.blockNumber.astype(int)
trans['value_ether'] = trans.value / 1e18
trans['gasPrice'] = trans.gasPrice.astype(float)
trans['gasUsed'] = trans.gasUsed.astype(float)
trans['value_cum'] = trans.value_ether.cumsum()
trans.cumulativeGasUsed = trans.cumulativeGasUsed.astype(float)
trans = trans.set_index('timeStamp')

dao_end_time = datetime(2016, 5, 28, 9, 0, 0, 0)
dao_phase_one_end = dao_end_time - timedelta(days=14)
dao_phase_two_end = dao_end_time - timedelta(days=5)

creation = trans[trans.index <= dao_end_time]
movement = trans[trans.index > dao_end_time]

exchanges = dict(
    poloniex = '0xb794f5ea0ba39494ce839613fffba74279579268',
    polo_2 = '0x32be343b94f860124dc4fee278fdcbd38c102d88',
    polo_new = '0xDf21fA922215B1a56f5a6D6294E6E36c85A0Acfb',
    kraken = '0x2910543af39aba0cd09dbb2d50200b3e800a63d2',
    bitfinex = '0xcafb10ee663f465f9d10588ac44ed20ed608c11e',  
    gatecoin_hacked = '0x40b9b889a21ff1534d018d71dc406122ebcf3f5a',
)

curator_contract_address = '0xda4a4626d3e16e094de3225a751aab7128e96526'

curators = {
'Vitalik Buterin':'1db3439a222c519ab44bb1144fc28167b4fa6ee6',
'Aeron Buchanan':'b2c1b92f4bed7a173547cc601fb73a1254d10d26',
'Christian Reitwießner':'0029218e1dab069656bfb8a75947825e7989b987',
'Taylor Gerring':'cee96fd34ec793b05ee5b232b0110eac0cc3327e',
'Viktor Tron':'b274363d5971b60b6aca27d6f030355e9aa2cf23',
'Fabian Vogelsteller':'c947faed052820f1ad6f4dda435e684a2cd06bb4',
'Martin Becze':'ae90d602778ed98478888fa2756339dd013e34c1',
'Gustav Simonsson':'e578fb92640393b95b53197914bd560b7bc2aac8',
'Vlad Zamfir':'127ac03acfad15f7a49dd037e52d5507260e1425',
'Gavin Wood':'0037a6b811ffeb6e072da21179d11b1406371c63',
'Alex Van de Sande':'d1220a0cf47c7b9be7a2e6ba89f429762e7b9adb',
'Iuri Matias':'C157f767030B4cDd1f4100e5Eb2b469b688D293E',
'Shermin Voshmgir':'820c6da74978799d574f61b01f8b5eebc051f95e',
'Gavin Wood':'0037a6b811ffeb6e072da21179d11b1406371c63',
}

function_codes_str = [
'643f7cdd=DAOpaidOut',
'82bf6464=DAOrewardAccount',
'39d1f908=actualBalance',
'dd62ed3e=allowance',
'4df6d6cc=allowedRecipients',
'095ea7b3=approve',
'70a08231=balanceOf',
'e5962195=blocked',
'749f9889=changeAllowedRecipients',
'e33734fd=changeProposalDeposit',
'eceb2945=checkProposalCode',
'4b6753bc=closingTime',
'baac5300=createTokenProxy',
'e66f53b7=curator',
'149acf9a=daoCreator',
'1f2dc5ef=divisor',
'237e9492=executeProposal',
'21b5b8dd=extraBalance',
'cc9ae3f6=getMyReward',
'be7c29c1=getNewDAOAddress',
'78524b2e=halveMinQuorum',
'b7bc2c84=isFueled',
'96d7f3f5=lastTimeMinQuorumMet',
'674ed066=minQuorumDivisor',
'0c3b7b96=minTokensToCreate',
'6837ff1e=newContract',
'612e45a3=newProposal',
'8d7af473=numberOfProposals',
'81f03fcb=paidOut',
'f8c80d26=privateCreation',
'8b15a605=proposalDeposit',
'013cf08b=proposals',
'a3912ec8=receiveEther',
'590e1ae3=refund',
'a1da2fb9=retrieveDAOReward',
'0e708203=rewardAccount',
'cdef91d0=rewardToken',
'82661dc4=splitDAO',
'34145808=totalRewardToken',
'18160ddd=totalSupply',
'a9059cbb=transfer',
'23b872dd=transferFrom',
'dbde1988=transferFromWithoutReward',
'4e10c3ee=transferWithoutReward',
'2632bf20=unblockMe',
'c9d27afe=vote',
]

dao_fn_codes = dict()
for string in function_codes_str:
    code, function = string.split('=')
    dao_fn_codes[function] = str(code)

def check_for_function(string: str, function: str):
    called = []
    for key, value in dao_fn_codes.items():
        if value in string:
            called.append(key)
    return ''.join(called)
    
    
trans['functions'] = [check_for_function(string, dao_fn_codes) for string in trans['input'].values]
trans.head()

trans[trans['functions'].isin(['splitDAO'])]

def token_rate(timeStamp):
    token_rate = [1.0 + 0.05*(x+1) for x in range(10)]
    days = [datetime(2016, 5, 28, 9, 0, 0, 0) - timedelta(days=13) + timedelta(days=1)*x for x in range(10)]
    
    for rate, day in zip(token_rate,days):
        if timeStamp < days[0] - timedelta(days=1):
            return 1.0
        elif day < timeStamp < day + timedelta(days=1):
            return rate
    return 1.5

creation['token_rate'] = creation.reset_index().timeStamp.apply(token_rate)
creation['tokens'] = creation.value_ether / creation.token_rate * 100

cum_ether = pd.DataFrame(creation.groupby('from').value_ether.sum().reset_index(), columns=['from', 'value_ether'])
cum_tokens = pd.DataFrame(creation.groupby('from').tokens.sum().reset_index(), columns=['from','tokens'])
cum_trans = creation.groupby('from').size().sort_values().reset_index()
cum_trans.columns = ['from', 'num_trans']
grouped = pd.merge(cum_ether, cum_trans, on='from')
grouped = pd.merge(grouped, cum_tokens, on='from')

# pd.merge(trans[trans['isSplit']][['timeStamp', 'from']], grouped, on='from')

creation[creation['functions'].isin(['transfer'])]



