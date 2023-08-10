from web3 import Web3, HTTPProvider, IPCProvider

gethRPCUrl='http://localhost:8545'
web3 = Web3(HTTPProvider(gethRPCUrl))

# Retrieve the last block number available from geth RPC
currentblock = web3.eth.getBlock('latest').number
print("Latest block: " + str(currentblock))

from hexbytes import HexBytes
def getBlockRange(blockstart,blockend):
    blocksDict = [ ]  
    for block in range(blockstart,blockend):  
        blocksDict.append(dict(web3.eth.getBlock(block,full_transactions=False)))
    return blocksDict

def getTransactionsInRange(blockstart,blockend):
    transactions_in_range=[]
    for block in range(blockstart,blockend):       
        transactions_in_block = web3.eth.getBlock(block,full_transactions=True)['transactions']       
        for transaction in transactions_in_block:
            cleansesed_transactions=(dict(transaction))
            cleansesed_transactions['blockHash'] = HexBytes(transaction['blockHash']).hex()
            cleansesed_transactions['hash'] = HexBytes(transaction['hash']).hex()
            transactions_in_range.append(cleansesed_transactions)
    return transactions_in_range

TXS_IN_RANGE=getTransactionsInRange(currentblock-1,currentblock)

TXS_IN_RANGE[0]

def getAllEventLogs(blockstart,blockend):
    tx_event_logs = [ ]
    for transaction in getTransactionsInRange(blockstart,blockend):
        tx_event=dict(web3.eth.getTransactionReceipt(transaction_hash=transaction['hash']))
        if(tx_event is not None):
            if(tx_event['logs'] is not None and tx_event['logs']):
                # Create a new santized_logs json
                santized_logs = [ ]
                for event_log in tx_event['logs']:
                    # AttributeDict -> Dict
                    santized_logs.append(dict(event_log))
                tx_event['logs'] = santized_logs
                # HexBytes -> String
                tx_event['transactionHash'] = HexBytes(tx_event['transactionHash']).hex()
            
                tx_event_logs.append(dict(tx_event))
    return tx_event_logs

EVENTS_WITH_LOGS=getAllEventLogs(currentblock-1,currentblock)

# Sample event with 1 method producing logs

EVENTS_WITH_LOGS[0]

# Sample event with 2 method invocations within 1 contract, producing logs[] length 2.
EVENTS_WITH_LOGS[2]



